#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace mlir;

/**
* Fill some known values to the mlir and use existing passes to do constant propagation.
* Including:
* - CSE: Common Subexpression Elimination
* - Canonlicalize
* - SCCP: Sparse Conditional Constant Propagation
* Ref: https://mlir.llvm.org/docs/Passes/
*/
static void preprocWithExistingPasses(PassManager& pm, func::FuncOp funcOp) {
  // First replace some known constants to the mlir
  
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

// ShapeInference by tracking constant number
void runShapeInference(MLIRContext& context, mlir::ModuleOp moduleOp){
  mlir::PassManager pm(&context);
  moduleOp->walk([&](func::FuncOp funcOp){
    preprocWithExistingPasses(pm, funcOp);
  });
  return;
}

