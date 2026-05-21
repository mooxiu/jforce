#include "../support/utilities.h"
#include "Utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

static bool foldLoadOpWithConstant(OpBuilder &opBuilder, fir::LoadOp loadOp,
                                   int constantVal) {
  opBuilder.setInsertionPoint(loadOp);
  auto resValue = loadOp.getResult();
  auto resType = resValue.getType();
  if (!llvm::isa<mlir::IntegerType>(resType) &&
      !llvm::isa<mlir::IndexType>(resType)) {
    return false;
  }

  arith::ConstantIntOp cop = arith::ConstantIntOp::create(
      opBuilder, loadOp.getLoc(), resValue.getType(), constantVal);
  loadOp.replaceAllUsesWith(cop.getResult());
  assert(loadOp.use_empty() && "Still been used!");
  return true;
}

struct PropagateConstantsPass
    : public mlir::PassWrapper<PropagateConstantsPass,
                               mlir::OperationPass<func::FuncOp>> {

  StringRef getArgument() const override {
    return "jforce-propagate-constants";
  }

  /// Fill some known values to the mlir and use existing passes to do constant
  /// propagation. Including:
  /// - CSE: Common Subexpression Elimination
  /// - Canonlicalize
  /// - SCCP: Sparse Conditional Constant Propagation
  /// Ref: https://mlir.llvm.org/docs/Passes/

  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());
    llvm::DenseMap<Value, int> valueMap;

    // some parameters containing the shape info are passed as pointer like
    for (int i = 0; i < funcOp.getNumArguments(); i++) {
      auto argType =
          funcOp.getArgAttrOfType<IntegerAttr>(i, JIT_ARG_TYPE_NAME_ATTR);
      if (argType && (argType.getValue() == ArgType::SHAPE_OR_BOUND)) {
        auto intAttr = funcOp.getArgAttrOfType<mlir::IntegerAttr>(
            i, JIT_LITERAL_VAL_ATTR_NAME);
        if (intAttr) {
          // `intAttr` is the literal address, need to recover to specific
          // number.
          auto argTy = funcOp.getArgumentTypes()[i];
          auto eleTy = getDTypeFromValueType(argTy);
          assert(eleTy == DType::I32 && "Supposed to be shape size!\n");
          auto eleVal = extractLiteralPtr(intAttr.getInt(), eleTy);
          assert(eleVal.returnedType == DType::I32);
          valueMap.insert(
              std::pair<Value, int>(funcOp.getArgument(i), eleVal.valI32));
        }
      }
    };

    auto getSolidVal = [&](Value v) -> std::pair<int, bool> {
      auto it = valueMap.find(v);
      if (it != valueMap.end()) {
        return std::pair(it->getSecond(), true);
      }
      return std::pair(-1, false);
    };

    // Replace some known values with constant values, then lifiting the
    // propagation task to existing mlir passes.
    llvm::SmallVector<Operation *> opsToDelete;
    funcOp.walk([&](fir::LoadOp lop) {
      auto lopVal = getSolidVal(lop.getOperand());
      if (lopVal.second) {
        if (foldLoadOpWithConstant(opBuilder, lop, lopVal.first)) {
          opsToDelete.push_back(lop);
        }
      }
    });
    funcOp.walk([&](hlfir::DeclareOp declareOp) {
      auto declaredVal = getSolidVal(declareOp.getMemref());
      if (declaredVal.second) {
        // This declareOp declares a constant value, and it will be used as
        // shape or bound. I should find the usages of it and change to the
        // value.
        for (auto user : declareOp.getResult(0).getUsers()) {
          if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(user)) {
            if (foldLoadOpWithConstant(opBuilder, loadOp, declaredVal.first)) {
              opsToDelete.push_back(loadOp);
            }
          }
        }
      }
    });

    for (auto *op : opsToDelete) {
      op->erase();
    }
  }
};
} // namespace

namespace xla_jit {
std::unique_ptr<mlir::Pass> createPropagateConstantsPass() {
  return std::make_unique<PropagateConstantsPass>();
}

void registerPropagateConstantsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createPropagateConstantsPass();
  });
};
} // namespace xla_jit
