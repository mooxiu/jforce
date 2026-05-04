#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include "Utils.h"

using namespace mlir;

/// Objective:
/// Before:
///   %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
///   %10 = fir.do_loop %arg9 = %c1 to %8 step %c1 iter_args(%arg10 = %9) -> (i32) {
///     ...
///     %14 = hlfir.designate %4#0 (%13)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
///     %15 = fir.load %14 : !fir.ref<f64>
///     ...
///
/// After:
///   %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
///   %memref = builtin.cast %4#1 : <>
///   %10 = fir.do_loop %arg9 = %c1 to %8 step %c1 iter_args(%arg10 = %9) -> (i32) {
///     ...
///     %15 = affine.load %memref: memref<f64>
///     ...

namespace {
  static AffineExpr recursivelyBuildAffineExpr(Value idx, OpBuilder opBuilder, llvm::SmallVector<Value>& dims) {
    while (!idx.getType().isIndex()) {
      auto defOp = idx.getDefiningOp();
      auto convertOp = llvm::dyn_cast<fir::ConvertOp>(defOp);
      if (!convertOp) {
        llvm::dbgs() << "The Definition Op is not convertOp!\n";
        std::exit(EXIT_FAILURE);
      }
      idx = convertOp.getValue();
    }

    // If idx is a result of constantOp here:
    if (llvm::isa<arith::ConstantIndexOp>(idx.getDefiningOp())) {
      auto constOp = llvm::cast<arith::ConstantIndexOp>(idx.getDefiningOp());
      auto res = llvm::cast<IntegerAttr>(constOp.getValue());
      return opBuilder.getAffineConstantExpr(res.getInt());  
    }

    auto defOp = idx.getDefiningOp(); 
    if (!defOp) {
      auto it = std::find(dims.begin(), dims.end(), idx);
      if (it != dims.end()) {
        return opBuilder.getAffineDimExpr(std::distance(dims.begin(), it));
      } else {
        unsigned dimId = dims.size();
        dims.push_back(idx);
        return opBuilder.getAffineDimExpr(dimId);
      }
    }

    // defOp is one of arith result      
    if (auto addOp = llvm::dyn_cast<arith::AddIOp>(defOp)) {
      return recursivelyBuildAffineExpr(addOp.getLhs(), opBuilder, dims) 
        + recursivelyBuildAffineExpr(addOp.getRhs(), opBuilder, dims);
    } else if (auto subOp = llvm::dyn_cast<arith::SubIOp>(defOp)) {
      return recursivelyBuildAffineExpr(subOp.getLhs(), opBuilder, dims)
        - recursivelyBuildAffineExpr(subOp.getRhs(), opBuilder, dims);
    } else if (auto mulOp = llvm::dyn_cast<arith::MulIOp>(defOp)) {
      // One of them has to be constantOp
      auto lhsExpr = recursivelyBuildAffineExpr(mulOp.getLhs(), opBuilder, dims);
      auto rhsExpr = recursivelyBuildAffineExpr(mulOp.getRhs(), opBuilder, dims);
      if (llvm::isa<AffineConstantExpr>(lhsExpr) || llvm::isa<AffineConstantExpr>(rhsExpr)) {
        return lhsExpr * rhsExpr;
      } else {
        // This is not an affine 
        return nullptr;
      }
    } else {
      // this is not affine
      return nullptr; 
    }
  }

  struct FIRLoadToAffineLoadPass: 
    public mlir::PassWrapper<FIRLoadToAffineLoadPass, mlir::OperationPass<func::FuncOp>> {
    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<affine::AffineDialect>();
    }
    
    StringRef getArgument() const override {
      return "jforce-fir-load-to-affine-load";
    }

    void runOnOperation() override {
      auto op= getOperation();
      OpBuilder opBuilder(op.getContext());

      llvm::SmallVector<affine::AffineForOp> forOps; 
      op.walk([&](affine::AffineForOp forOp){forOps.push_back(forOp);});
      
      for (auto forOp: forOps) {
        llvm::SetVector<Operation*> toDeleteOps;
        forOp.walk([&](hlfir::DesignateOp designateOp){
          auto users = designateOp->getUsers();

          llvm::SmallVector<fir::LoadOp> loadOpUsers;
          for (auto user: users) {
            auto loadOpUser = llvm::dyn_cast<fir::LoadOp>(user); 
            if (!loadOpUser) {
              return;
            }
            loadOpUsers.push_back(loadOpUser);
          }

          auto sliceDeclareOp = llvm::dyn_cast<hlfir::DeclareOp>(designateOp.getMemref().getDefiningOp()); 
          if (!sliceDeclareOp) {
            llvm::dbgs() << "Definition of Slice is not a hlfir::declareOp";
            return;
          }
          
          opBuilder.setInsertionPointAfter(sliceDeclareOp);
          // static UnrealizedConversionCastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
          auto memrefTypeInfo = inspectTypeInfo(sliceDeclareOp.getResultTypes()[0]); 
          auto memrefType = MemRefType::get(memrefTypeInfo.shape, memrefTypeInfo.elementTy, {}, {});
          llvm::ArrayRef<NamedAttribute> attributes; 
          auto castOp = mlir::UnrealizedConversionCastOp::create(
            opBuilder, 
            op.getLoc(),
            {memrefType},
            {sliceDeclareOp.getResult(0)},
            attributes
          );
          
          opBuilder.setInsertionPoint(designateOp);
          llvm::SmallVector<Value> dims;
          llvm::SmallVector<Value> indices;
          for (Value idx: designateOp.getIndices()) {
            auto affineExpr = recursivelyBuildAffineExpr(idx, opBuilder, dims);
            if (affineExpr == nullptr) {
              // this is not affine, just return
              return;
            } 
            
            if (dims.empty()) {
              // Pure constant      
              int64_t constResult = llvm::cast<AffineConstantExpr>(affineExpr).getValue();
              Value constIdx = arith::ConstantIndexOp::create(opBuilder, op.getLoc(), constResult);
              indices.push_back(constIdx);
            } else {
              // Create map, apply...
              auto map = AffineMap::get(dims.size(), 0, affineExpr);
              auto finalIdx = affine::AffineApplyOp::create(opBuilder, op.getLoc(), map, dims).getResult();
              indices.push_back(finalIdx); 
            }
          }

          for (auto loadOpUser: loadOpUsers) {
            opBuilder.setInsertionPoint(loadOpUser);
            // static AffineLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value memref, ValueRange indices = {});
            assert(designateOp.getIndices().size() == 1);
            auto affineLoadOp = affine::AffineLoadOp::create(opBuilder, op.getLoc(), castOp.getResult(0), indices);
            loadOpUser.getResult().replaceAllUsesWith(affineLoadOp.getResult());
            toDeleteOps.insert(loadOpUser);   
          }
          toDeleteOps.insert(designateOp);
        }); 
        for (auto toDeleteOp: toDeleteOps) {
          toDeleteOp->erase();
        }
      }
    };  
  };
};

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createFIRLoadToAffineLoadPass() {
    return std::make_unique<FIRLoadToAffineLoadPass>();
  }

  void registerFIRLoadToAffineLoadPass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createFIRLoadToAffineLoadPass();});
  };
}

