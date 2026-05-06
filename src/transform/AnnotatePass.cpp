#include "../support/profiler.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
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

namespace {

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


// Do BFS. This might lead to compromised performance as it will loop until the
// end.
// TODO: We actually only need to know if it is used as a shape or (lowerbound
// || upperbound || step of a loop)
static ArgType setArgType(func::FuncOp funcOp, Value arg) {
  bool usedAsShapeOrBound = false;
  bool usedAsCompute = false;

  llvm::SmallVector<mlir::Value> worklist;
  llvm::DenseSet<mlir::Value> visited;

  worklist.push_back(arg);
  visited.insert(arg);
  while (!worklist.empty()) {
    mlir::Value val = worklist.pop_back_val();

    for (mlir::Operation *user : val.getUsers()) {
      if (llvm::isa<hlfir::DeclareOp, fir::LoadOp, fir::ConvertOp>(user)) {
        for (mlir::Value res : user->getResults()) {
          if (visited.insert(res).second) {
            worklist.push_back(res);
          }

        }
        continue;
      }

      if (auto loopOp = llvm::dyn_cast<fir::DoLoopOp>(user)) {
        if (llvm::is_contained(
            {loopOp.getStep(), loopOp.getLowerBound(), loopOp.getUpperBound()}, 
            val)) {
          return ArgType::SHAPE_OR_BOUND;
        }
      }  
    }
  }
  return ArgType::OTHER;
}

struct AnnotatePass
    : public PassWrapper<AnnotatePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotatePass)

  StringRef getArgument() const override { return "jforce-annotate"; }
  void runOnOperation() override {
    PROFILE_SCOPE("annotate compute args", Phase::LOWERING_EXTRA);
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());

    for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
      auto arg = funcOp.getArgument(i);
      funcOp.setArgAttr(i, JIT_ARG_TYPE_NAME_ATTR,
                        opBuilder.getUI32IntegerAttr(setArgType(funcOp, arg)));
    }
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
