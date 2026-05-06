#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <memory>

using namespace mlir;

/// The objective of this pass is to move affine for loop or other blocks in a separate function,
/// so the Enzyme-JAX's AffineToStableHLO pass can work. 
namespace {

static void outlineAffineForOp(ModuleOp moduleOp, func::FuncOp funcOp, OpBuilder& opBuilder, affine::AffineForOp forOp) {
  // Collect all values defined outside of the affineOp itself.
  llvm::SetVector<Value> outDefinedVals;
  mlir::getUsedValuesDefinedAbove({forOp.getRegion()}, outDefinedVals);
  for (auto operand: forOp.getOperands()) {
    outDefinedVals.insert(operand);
  }
  llvm::SmallVector<Type> outlinedFuncInputTypes;
  for (auto val: outDefinedVals) {
    outlinedFuncInputTypes.push_back(val.getType());
  }
  opBuilder.setInsertionPoint(funcOp);
  auto outlinedFuncType = opBuilder.getFunctionType(outlinedFuncInputTypes, {});
  auto outlinedFuncName = llvm::formatv("outlined_affinefor_{0}", reinterpret_cast<std::uintptr_t>(forOp.getAsOpaquePointer())).str();
  llvm::SmallVector<NamedAttribute> attrs = {};
  llvm::SmallVector<DictionaryAttr> argAttrs = {};
  auto outlinedFunc = func::FuncOp::create(opBuilder, funcOp.getLoc(), outlinedFuncName, outlinedFuncType, attrs, argAttrs);

  // Copy from old to new
  auto entryBlock = outlinedFunc.addEntryBlock();
  opBuilder.setInsertionPointToEnd(entryBlock);
  IRMapping mapping;
  for (int i = 0; i < outDefinedVals.getArrayRef().size(); i++) {
    mapping.map(*(outDefinedVals.begin()+i), entryBlock->getArgument(i));
  }
  opBuilder.clone(*forOp.getOperation(), mapping);
  func::ReturnOp::create(opBuilder, outlinedFunc.getLoc()); // the returnOp should be empty

  // add Call function
  opBuilder.setInsertionPoint(forOp);
  func::CallOp::create(opBuilder, forOp.getLoc(), outlinedFunc, outDefinedVals.getArrayRef());
}

static void outlineAffineConstructs(ModuleOp moduleOp, func::FuncOp funcOp, OpBuilder& opBuilder) {
  llvm::SmallVector<Operation*> toDelete;
  funcOp.walk([&](affine::AffineForOp affineForOp){
    // creating a function, which has the inputs for all the slices and affine bounds. 
    outlineAffineForOp(moduleOp, funcOp, opBuilder, affineForOp);     
    toDelete.push_back(affineForOp);
    return;
  });   
  for (auto* op: toDelete) {
    op->erase();
  }
  return;
}

struct OutlineAffinePass:
  public mlir::PassWrapper<OutlineAffinePass, mlir::OperationPass<mlir::ModuleOp>> {
    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<affine::AffineDialect>();
      return;
    }

    StringRef getArgument() const override { return "jforce-outline-affine"; }

    void runOnOperation() override {
      auto moduleOp = getOperation();
      OpBuilder opBuilder(moduleOp.getContext());
      llvm::SmallVector<func::FuncOp> funcOps;
      moduleOp.walk([&](func::FuncOp funcOp){
        funcOps.push_back(funcOp);
      });
      for (auto funcOp: funcOps) {
        outlineAffineConstructs(moduleOp, funcOp, opBuilder);
      } 
    } 
  };
}

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createOutlineAffinePass() {
    return std::make_unique<OutlineAffinePass>();
  }

  void registerOutlineAffinePass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createOutlineAffinePass();});
  };
}

