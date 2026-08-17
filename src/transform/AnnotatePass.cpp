#include "Utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>

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

// TODO: may should extend to other loop type
static llvm::DenseSet<Value> markShapeOperands(func::FuncOp funcOp) {
  llvm::DenseSet<Value> shapeOperandsSet;
  auto insertOptional = [&](auto bounds) {
    if (!bounds) {
      return;
    }

    for (OpFoldResult bound : *bounds) {
      if (auto value = dyn_cast<Value>(bound))
        shapeOperandsSet.insert(value);
    }
  };

  funcOp.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<fir::ShapeOp, fir::ShapeShiftOp>(
            [&](auto) { shapeOperandsSet.insert_range(op->getOperands()); })
        .Case<fir::DoLoopOp>([&](fir::DoLoopOp lop) {
          shapeOperandsSet.insert(lop.getLowerBound());
          shapeOperandsSet.insert(lop.getUpperBound());
          shapeOperandsSet.insert(lop.getStep());
        })
        .Case<scf::ForOp>([&](scf::ForOp fop) {
          insertOptional(fop.getLoopLowerBounds());
          insertOptional(fop.getLoopUpperBounds());
          insertOptional(fop.getLoopSteps());
        })
        .Case<scf::ParallelOp>([&](scf::ParallelOp pop) {
          insertOptional(pop.getLoopLowerBounds());
          insertOptional(pop.getLoopUpperBounds());
          insertOptional(pop.getLoopSteps());
        })
        .Default([](auto) {});
  });
  return shapeOperandsSet;
}

// WARNING: following situation, we can not say  %arg0 contribute to the value
// of %shape0, but it should not happen in real practice? %1 = alloca i32
// %arg0Val = load %arg0
// store %arg0Val to %1
// store %c1 to %1
// %shape0 = load %1
// fir.loop %iv = %shape0, %shape1, %shape2
static void annotateArgs(func::FuncOp funcOp, OpBuilder &opBuilder,
                         const llvm::DenseSet<Value> &shapeOperands) {
  UnitAttr unitAttr = opBuilder.getUnitAttr();
  llvm::SmallVector<Value> worklist(shapeOperands.begin(), shapeOperands.end());

  llvm::DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second) {
      continue;
    }

    if (Operation *defOp = value.getDefiningOp()) {
      if (auto loadOp = dyn_cast<fir::LoadOp>(defOp)) {
        Value addr = loadOp.getMemref();

        for (Operation *user : addr.getUsers()) {
          if (auto storeOp = dyn_cast<fir::StoreOp>(user))
            worklist.push_back(storeOp.getValue());
        }
      }
      worklist.append(defOp->operand_begin(), defOp->operand_end());
      continue;
    }

    if (auto blockArg = llvm::dyn_cast<BlockArgument>(value)) {
      if (blockArg.getOwner() != &funcOp.front()) {
        continue;
      }
      CHECK_TYPE(blockArg.getType());
      funcOp.setArgAttr(blockArg.getArgNumber(), JIT_SHAPE_ARG_ATTR_NAME,
                        unitAttr);
    }
  }
}

struct AnnotatePass
    : public PassWrapper<AnnotatePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotatePass)

  StringRef getArgument() const override { return "jforce-annotate"; }
  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());

    auto shapeOperands = markShapeOperands(funcOp);
    for (auto op: shapeOperands) {
      llvm::dbgs() << "\n[DEBUG]: the op: ";
      op.printAsOperand(llvm::dbgs(), {});
    }
    annotateArgs(funcOp, opBuilder, shapeOperands);
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
