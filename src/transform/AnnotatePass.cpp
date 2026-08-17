#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstddef>
#include "Utils.h"

using namespace mlir; 

static void checkArgIsInt(Type type) {
  auto refType = llvm::dyn_cast<fir::ReferenceType>(type);
  assert(refType && "Argument type is supposed to be reference type!");
  assert(refType.getElementType().isIntOrIndex());
}

#ifdef ENABLE_XLA_DEBUG
  #define CHECK_TYPE(type) checkArgIsInt(type)
#else
  #define CHECK_TYPE(type) ((void)0)
#endif

namespace {

struct AnnotatePass
    : public PassWrapper<AnnotatePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotatePass)

  // Backtracking from fir::Shape and fir::ShapeShiftOp, if an argument has been used in the track, mark it as SHAPE_ARG.
  static void annotateArgs(func::FuncOp funcOp, OpBuilder& opBuilder) {
    llvm::DenseSet<Value> visited;
    llvm::SmallVector<Value> shapesOperands;
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
      assert(!llvm::isa<fir::DeclareOp>(op));
    });
    size_t next = 0;
    while (next < shapesOperands.size()) {
      Value val = shapesOperands[next++];
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
        if (blockArg.getOwner() == &funcOp.front()) {
          // WARN: There might be edge cases like:
          // ```
          // float arg;
          // DO i = 1, floor(arg)
          //    Y(i) = X(i)
          // END DO
          // ```
          // but let me pretend they do not exist for now.
          CHECK_TYPE(blockArg.getType());
          funcOp.setArgAttr(
            blockArg.getArgNumber(),
            JIT_SHAPE_ARG_ATTR_NAME,
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
  }

  StringRef getArgument() const override { return "jforce-annotate"; }
  void runOnOperation() override {
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
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createAnnotatePass(); });
};
} // namespace xla_jit
