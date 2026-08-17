#include "../support/utilities.h"
#include "Utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>

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

static int retrieveConstVal(Type argType, IntegerAttr intAttr) {
  auto eleTy = getDTypeFromValueType(argType);
  assert((eleTy == DType::I32 || eleTy == DType::I64) &&
         "Supposed to be shape size!\n");
  auto eleVal = extractLiteralPtr(intAttr.getInt(), eleTy);
  assert(eleVal.valI32 == 0 || eleVal.valI64 == 0);
  assert(eleVal.valI32 > 0 || eleVal.valI32 > 0);
  int constVal; 
  if (eleVal.valI32 > 0) {
    constVal = eleVal.valI32;
  } else if (eleVal.valI64 > 0) {
    constVal = eleVal.valI64;
  }
  return constVal;
}

static llvm::SmallVector<fir::LoadOp> collectLoadFromShapeVal(
  Value arg
) {
  llvm::SmallVector<fir::LoadOp> loadOpsToMaterialize;
  for (Operation *user : arg.getUsers()) {
    if (!llvm::isa<fir::LoadOp, hlfir::DeclareOp>(user)) {
      continue;
    }
    if (auto lop = llvm::dyn_cast<fir::LoadOp>(user)) {
      loadOpsToMaterialize.push_back(lop);
    }
    if (auto declareOp = llvm::dyn_cast<hlfir::DeclareOp>(user)) {
      for (auto user : declareOp.getResult(0).getUsers()) {
        if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(user)) {
          loadOpsToMaterialize.push_back(loadOp);
        }
      }
    }
  }
  return loadOpsToMaterialize;
}


static void materializeShapeArgs(func::FuncOp funcOp, OpBuilder &opBuilder) {
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    if (funcOp.getArgAttrOfType<UnitAttr>(i, JIT_SHAPE_ARG_ATTR_NAME)) {
      auto intAttr =
          funcOp.getArgAttrOfType<IntegerAttr>(i, JIT_LITERAL_VAL_ATTR_NAME);
      if (!intAttr) {
        llvm::dbgs() << "\n[DEBUG] Include dynamic shape!\n";
        continue;
      }
      int constVal = retrieveConstVal(funcOp.getArgumentTypes()[i], intAttr);
      auto loadOpsToMaterialize = collectLoadFromShapeVal(funcOp.getArgument(i));
      if (loadOpsToMaterialize.empty()) continue;
      
      // Start to materialize
      for (auto loadOp: loadOpsToMaterialize) {
        foldLoadOpWithConstant(opBuilder, loadOp, constVal);
        loadOp.erase();
      }
    }
  }
}

struct PropagateConstantsPass
    : public mlir::PassWrapper<PropagateConstantsPass,
                               mlir::OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "jforce-propagate-constants";
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());
    materializeShapeArgs(funcOp, opBuilder);
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
