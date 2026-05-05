#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "../support/profiler.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstddef>

using namespace mlir; 

#define JIT_COMPUTE_ARG_ATTR_NAME "jit.compute_arg"
#define JIT_SHAPE_META_ATTR_NAME "jit.shape_meta"

namespace {


struct AnnotatePass: 
  public PassWrapper<AnnotatePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotatePass)
  
  StringRef getArgument() const override { 
    return "jforce-annotate"; 
  }

  static void annotateArgs(func::FuncOp funcOp, OpBuilder& opBuilder) {
    llvm::DenseSet<Value> visited;
    llvm::SmallVector<Value> shapesOperands;
    llvm::SmallVector<bool> computeArgs(funcOp.getNumArguments(), false);
    UnitAttr unitAttr = opBuilder.getUnitAttr();

    funcOp.walk([&](Operation* op){
      if (llvm::isa<fir::ShapeOp, fir::ShapeShiftOp>(op)){
        llvm::for_each(op->getOperands(), [&](Value operand){
          if (!visited.contains(operand)) {
            shapesOperands.push_back(operand);
            visited.insert(operand);
          }
        });
      }
      if (auto declareOp = llvm::dyn_cast<hlfir::DeclareOp>(op)) {
        Value mem = declareOp.getMemref();
        if (auto blockArg = llvm::dyn_cast<BlockArgument>(mem);
          blockArg && blockArg.getOwner() == &funcOp.front()) {
          computeArgs[blockArg.getArgNumber()] = true;
        }
      }
      assert(!llvm::isa<fir::DeclareOp>(op));
    });
    size_t next = 0;
    while (next < shapesOperands.size()) {
      Value val = shapesOperands[next++];
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
        if (blockArg.getOwner() == &funcOp.front()) {
          funcOp.setArgAttr(
            blockArg.getArgNumber(),
            JIT_SHAPE_META_ATTR_NAME,
            unitAttr 
          );
        }
        continue;
      }

      Operation *definingOp = val.getDefiningOp();
      if (!definingOp) continue;

      for (Value operand : definingOp->getOperands()) {
        if (visited.insert(operand).second) {
          shapesOperands.push_back(operand);
        }
      }
    }

    for (unsigned i = 0; i < computeArgs.size(); i++) {
      if (computeArgs[i]) {
        funcOp.setArgAttr(i, JIT_COMPUTE_ARG_ATTR_NAME, unitAttr);
      }
    }
  }

  void runOnOperation() override {
    PROFILE_SCOPE("annotate compute args", Phase::LOWERING_EXTRA);
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());

    annotateArgs(funcOp, opBuilder);
  }
};
} // namespace

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createAnnotatePass() {
    return std::make_unique<AnnotatePass>();
  }

  void registerAnnotatePass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createAnnotatePass();});
  };
}
