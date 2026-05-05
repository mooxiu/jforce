#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "Passes.h"

using namespace mlir;

/// Loop Invariant Code Motion for Fortran context.
/// Official `--loop-invariant-code-motion` is conservative but we can do more.
struct FirLICMPass: 
  public mlir::PassWrapper<FirLICMPass, mlir::OperationPass<mlir::func::FuncOp>> {
  
  // The return value could be a nullptr.
  static Operation* getOutermostLoopNotContainingDefOp(func::FuncOp funcOp, fir::LoadOp loadOp, Operation* defOp) {
    Operation* currLoop = nullptr;
    Operation* currOp = loadOp;
    while (currOp != funcOp && !currOp->isAncestor(defOp)) {
      if (llvm::isa<
                    affine::AffineForOp,
                    scf::ForOp,
                    fir::DoLoopOp
        >(currOp)) {
        currLoop = currOp;
      }
      currOp = currOp->getParentOp(); 
    }
    return currLoop;
  }

  static void moveFirLoad(func::FuncOp funcOp, OpBuilder& opBuilder) {
    llvm::SmallVector<fir::LoadOp> toDelete;
    funcOp.walk([&](fir::LoadOp loadOp){
      auto defOp = loadOp.getOperand().getDefiningOp(); 
      if (llvm::isa<hlfir::DeclareOp>(defOp)) {
        auto outermostLoop = getOutermostLoopNotContainingDefOp(funcOp, loadOp, defOp);
        if (outermostLoop != nullptr) {
          opBuilder.setInsertionPoint(outermostLoop);
          // static LoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, mlir::Value refVal);
          auto movedLoadOp = fir::LoadOp::create(opBuilder, outermostLoop->getLoc(), loadOp.getMemref());
          loadOp.getResult().replaceAllUsesWith(movedLoadOp.getResult());
          toDelete.push_back(loadOp);   
        } else {
          return;
        }
      } else {
        // Nothing we can do, just skip this pass.
        return;
      }
    });
    for (auto op: toDelete) {op.erase();}
  }

  StringRef getArgument() const override {
    return "jforce-fir-licm";
  }

  void runOnOperation() override {
    auto funcOp = getOperation();    
    OpBuilder opBuilder(funcOp.getContext());
    moveFirLoad(funcOp, opBuilder);
  } 
};

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createFirLICMPass() {
    return std::make_unique<FirLICMPass>();
  }

  void registerFirLICMPass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createFirLICMPass();});
  };
}

