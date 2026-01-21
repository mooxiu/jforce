#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstdint>
#include <memory>
#include <utility>

using namespace mlir; 

namespace {

struct StaticShapeOptimizationPass : public PassWrapper<StaticShapeOptimizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticShapeOptimizationPass);
public:
  class StaticShapeConverter : public TypeConverter {
  public:
    StaticShapeConverter() {
      // addConversion([&](fir::ReferenceType type) -> Type {});
    };
  };

  struct DeclareOpConversion: public OpConversionPattern<hlfir::DeclareOp> {
    using OpConversionPattern::OpConversionPattern; 
  };

  struct ElementalOpConversion: public OpConversionPattern<hlfir::ElementalOp> {
    using OpConversionPattern::OpConversionPattern; 
  };

  struct DesignateOpConversion: public OpConversionPattern<hlfir::DesignateOp> {
    using OpConversionPattern::OpConversionPattern; 
  };

  struct ApplyOpConversion: public OpConversionPattern<hlfir::ApplyOp> {
    using OpConversionPattern::OpConversionPattern; 
  };

  struct AssignOpConversion: public OpConversionPattern<hlfir::AssignOp> {
    using OpConversionPattern::OpConversionPattern; 
  };

  struct DestroyOpConversion: public OpConversionPattern<hlfir::DestroyOp> {
    using OpConversionPattern::OpConversionPattern; 
  };

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext* context = &getContext();
    // Analysis
    analysis(funcOp);   
    
    // Rewriting
    StaticShapeConverter converter;
    RewritePatternSet patterns(context);

    patterns.add<DeclareOpConversion>(converter, context);
    patterns.add<ElementalOpConversion>(converter, context);
    patterns.add<DesignateOpConversion>(converter, context);
    patterns.add<ApplyOpConversion>(converter, context);
    patterns.add<AssignOpConversion>(converter, context);
    patterns.add<DestroyOpConversion>(converter, context);

    ConversionTarget convTarget(*context);
    convTarget.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp funcOp){
      return converter.isSignatureLegal(funcOp.getFunctionType());  
    });

    if (failed(applyPartialConversion(funcOp, convTarget, std::move(patterns)))){
      signalPassFailure();
    };
    return;
  };

private:
  // mapping from value to shape (a vector of each dimension)
  llvm::DenseMap<Value, llvm::SmallVector<int64_t>> shapeMap;
  // tracking shape constant
  llvm::DenseMap<Value, int64_t> constTrackingMap;
    
  void analysis(func::FuncOp funcOp) {
    funcOp.walk([&](Operation* op){
      llvm::TypeSwitch<Operation*>(op)
        .Case<arith::ConstantOp>([&](arith::ConstantOp cop){
          auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cop.getValue());
          if (intAttr) {
            constTrackingMap.insert(std::pair<Value, int64_t>(cop.getResult(), intAttr.getInt())); 
          }
        })
        .Case<fir::ShapeOp>([&](fir::ShapeOp sop){
          // %0 = fir.shape %c1000, %c1000 : (index, index) -> !fir.shape<2>
          sop.getResult();

          llvm::SmallVector<int64_t> sizes;
          sizes.reserve(sop.getNumOperands());
          for (unsigned i = 0; i < sop.getNumOperands(); i++) {
            auto opr = sop.getOperand(i);
            assert(constTrackingMap.contains(opr) && "Operand static value should be known!");
            sizes.push_back(constTrackingMap.at(opr));
          }
        })
        .Default([](auto){});
    }); 
  } 
};


} // namespace

