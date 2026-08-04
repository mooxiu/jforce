#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "../support/profiler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
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

  static void markShapeArgs(func::FuncOp funcOp, OpBuilder& opBuilder) {
    llvm::DenseSet<Value> visited;
    llvm::SmallVector<Value> shapesOperands;
    funcOp.walk([&](Operation* op){
      if (llvm::isa<fir::ShapeOp, fir::ShapeShiftOp>(op)){
        llvm::for_each(op->getOperands(), [&](Value operand){
          if (!visited.contains(operand)) {
            shapesOperands.push_back(operand);
            visited.insert(operand);
          }
        });
      }
    });
    size_t next = 0;
    while (next < shapesOperands.size()) {
      Value val = shapesOperands[next++];
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
        if (blockArg.getOwner() == &funcOp.front()) {
          funcOp.setArgAttr(
            blockArg.getArgNumber(),
            JIT_SHAPE_META_ATTR_NAME,
            opBuilder.getUnitAttr()
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
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());

    markShapeArgs(funcOp, opBuilder);

    for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
      auto arg = funcOp.getArgument(i);
      bool isCompute = false;

      for (auto* user: arg.getUsers()) {
        if (llvm::isa<hlfir::DeclareOp>(user) || llvm::isa<fir::DeclareOp>(user)) {
          isCompute = true;
        } 
      }

      if (isCompute) {
        funcOp.setArgAttr(i, JIT_COMPUTE_ARG_ATTR_NAME, opBuilder.getUnitAttr());
      }
    }
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
