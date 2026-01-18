#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace mlir;

static llvm::DenseMap<Value, int> valueMap;

// If this is an argument for shape, then we return the positive int value;
// If not, return -1;
static int getSolidVal(Value v) {
  if (valueMap.contains(v)) {
    return valueMap.at(v);
  }
  return -1;  
}

/**
* Fill some known values to the mlir and use existing passes to do constant propagation.
* Including:
* - CSE: Common Subexpression Elimination
* - Canonlicalize
* - SCCP: Sparse Conditional Constant Propagation
* Ref: https://mlir.llvm.org/docs/Passes/
*/
static void preprocWithExistingPasses(OpBuilder opBuilder, PassManager& pm, func::FuncOp funcOp, llvm::DenseMap<int, int> constShapeMap) {
    // First replace some known constants to the mlir
  funcOp.walk([&](fir::LoadOp lop){
    opBuilder.setInsertionPoint(lop);
    if (getSolidVal(lop.getOperand()) > 0) {
      auto resValue = lop.getResult();
      arith::ConstantIntOp cop = arith::ConstantIntOp::create(opBuilder, funcOp.getLoc(), resValue.getType(), getSolidVal(lop.getOperand()));
      cop.print(llvm::dbgs());
      lop.replaceAllUsesWith(cop.getResult());
      assert(lop.use_empty() && "Still been used!");
      lop.erase();
    }
  });
  
  // Run passes
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSCCPPass());
  if (mlir::failed(pm.run(funcOp))){
    std::cerr << "[Fail] Fail to run passes on funcOp!" << std::endl;
    exit(EXIT_FAILURE);
  };
  return;
}

static void shapeInferenceInternal() {
  return;
}

// ShapeInference by tracking constant number
void runShapeInference(MLIRContext& context, mlir::ModuleOp moduleOp, llvm::DenseMap<int, int>& constShapeMap){
  std::cout << "--------------ConstShapeMap: \n";
  std::for_each(constShapeMap.begin(), constShapeMap.end(), [](std::pair<int, int> p){
      std::cout << "key: " << p.first << "; value: " << p.second << "\n";
  });
  std::cout << "\n--------------ConstShapeMap. \n";

  mlir::PassManager pm(&context);
  OpBuilder opBuilder(&context);

  moduleOp->walk([&](func::FuncOp funcOp){
    for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
      if (constShapeMap.contains(i)) {
        valueMap.insert(std::pair<Value, int>(funcOp.getArgument(i), constShapeMap[i]));
      }
    }
    preprocWithExistingPasses(opBuilder, pm, funcOp, constShapeMap);
  });

  std::cout << "--------------After shape inference:\n";
  moduleOp.print(llvm::dbgs());
  std::cout << "\n--------------After shape inference.\n";

  return;
}

