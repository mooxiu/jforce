#include "../support/profiler.h"
#include "Utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

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
      if (llvm::isa<hlfir::DeclareOp, fir::DeclareOp, fir::LoadOp, fir::ConvertOp,
          arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp, arith::DivUIOp>(user)) {
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
      } else if (llvm::isa<fir::ShapeOp, fir::ShapeShiftOp>(user)) {
        return ArgType::SHAPE_OR_BOUND;
      } else if (auto cmpIOp = llvm::dyn_cast<arith::CmpIOp>(user)) {
        if (cmpIOp.getOperand(0).getType().isIndex()) {
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
