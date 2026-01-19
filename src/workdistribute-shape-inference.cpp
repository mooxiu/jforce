#include "utilities.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
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
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (mlir::failed(pm.run(funcOp))){
    std::cerr << "[Fail] Fail to run passes on funcOp!" << std::endl;
    exit(EXIT_FAILURE);
  };
  return;
}



static void shapeInferenceInternal(OpBuilder opBuilder, func::FuncOp funcOp) {
  // mapping from value to shape (a vector of each dimension)
  llvm::DenseMap<Value, llvm::SmallVector<int64_t>> shapeMap;
  // tracking shape constant
  llvm::DenseMap<Value, int64_t> constTrackingMap;

  funcOp.walk([&](Operation* op){
    llvm::TypeSwitch<Operation*>(op)
      .Case<arith::ConstantOp>([&](arith::ConstantOp cop){
        auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cop.getValue());
        if (intAttr) {
          constTrackingMap.insert(std::pair<Value, int64_t>(cop.getResult(), intAttr.getInt())); 
        }
      })
      .Case<fir::ShapeOp>([&](fir::ShapeOp sop){
        // %0 = fir.shape %c1000, %c1000 : (index, index) -> !fir.shape<2>
        llvm::SmallVector<int64_t> sizes; // {1000, 1000} in this example
        sizes.reserve(sop.getNumOperands());
        for (unsigned i = 0; i < sop.getNumOperands(); i++) {
          auto opr = sop.getOperand(i);
          assert(constTrackingMap.contains(opr) && "Operand static value should be known!");
          sizes.push_back(constTrackingMap.at(opr));
        }
        shapeMap.insert(std::pair(sop.getResult(), sizes)); // %0 -> {1000, 1000}
      })
      .Case<hlfir::DeclareOp>([&](hlfir::DeclareOp dop){
        // Example: %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFcoexecute_aEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
        // Objective: %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFcoexecute_aEz"} : (!fir.ref<!fir.array<1000x1000xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<1000x1000xf64>>, !fir.ref<!fir.array<1000x1000xf64>>)
        llvm::dbgs() << "\n DeclareOp: \n";
        for (unsigned i = 0; i < dop.getNumResults(); i++) {
          llvm::dbgs() << "\n Result" << i << ": ";
          dop.getResult(i).printAsOperand(llvm::dbgs(), {});
          llvm::dbgs() << "\n";
        }
        assert(shapeMap.contains(dop.getShape()) && "Expect the shape of the decalreOp already known!");
        auto newShape = shapeMap.at(dop.getShape());
        opBuilder.setInsertionPoint(dop);

        auto ndop = hlfir::DeclareOp::create(
            opBuilder,
            dop.getLoc(),           
            convertToStaticShape(dop.getResult(0).getType(), newShape),
            convertToStaticShape(dop.getResult(1).getType(), newShape), 
            dop.getMemref(),        
            dop.getShape(),         
            dop.getTypeparams(),    
            dop.getDummyScope(),    
            dop.getStorage(),       
            dop.getStorageOffsetAttr(), 
            dop.getUniqNameAttr(),      
            dop.getFortranAttrsAttr(),  
            dop.getDataAttrAttr(),      
            nullptr,
            dop.getDummyArgNoAttr()     
        );
          // auto ndop = hlfir::DeclareOp::create(opBuilder, funcOp.getLoc(), newBoxType, dop.getMemref(), dop.getShape(), dop.getTypeparams(), dop.getDummyScope(), dop.getStorage(), dop.getStorageOffsetAttr(), dop.getUniqNameAttr(), dop.getFortranAttrsAttr(), dop.getDataAttrAttr(), dop.getDummyArgNoAttr());
        llvm::dbgs() << "\n Updated DeclaredOP: \n";
        ndop.print(llvm::dbgs(), {});
        llvm::dbgs() << "\n";
        dop.replaceAllUsesWith(ndop.getResults());
        dop.erase();
      })
      .Case<hlfir::DesignateOp>([&](hlfir::DesignateOp dop){
        // Example: %8 = hlfir.designate %2#0 (%arg10, %arg11)  : (!fir.box<!fir.array<?x?xf64>>, index, index) -> !fir.ref<f64>
      })
      .Case<hlfir::ElementalOp>([&](hlfir::ElementalOp eop){
        // Example: %6 = hlfir.elemental %0 unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf64> {
      })
      .Case<hlfir::AssignOp>([&](hlfir::AssignOp aop){
        // Example: hlfir.assign %7 to %1#0 : !hlfir.expr<?x?xf64>, !fir.box<!fir.array<?x?xf64>>
      })
      .Case<hlfir::DestroyOp>([&](hlfir::DestroyOp dop){
        // Example: hlfir.destroy %7 : !hlfir.expr<?x?xf64>
      })
      .Default([](auto){});
  }); 
  

  // TODO: Update the function arguments signature 
  //
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

    std::cout << "\n--------------ShapeInferenceInternal Log:\n";
    shapeInferenceInternal(opBuilder, funcOp);
    std::cout << "\n--------------ShapeInferenceInternal Log:\n";
  });

  std::cout << "--------------After shape inference:\n";
  moduleOp.print(llvm::dbgs());
  std::cout << "\n--------------After shape inference.\n";

  return;
}

