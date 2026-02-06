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
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>

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

// Not a roboust transformation but works for now.
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
        if (isDynamicShape(dop.getResult(0).getType())) {
          auto staticShape = shapeMap.at(dop.getShape());

          // propagate the shape of the results
          for (unsigned i = 0; i < dop.getNumResults(); i++) {
            shapeMap.insert(std::pair(dop.getResult(i), staticShape));
          }
          shapeMap.insert(std::pair(dop.getMemref(), staticShape));

          // Insert the new declareOp
          opBuilder.setInsertionPoint(dop);
          auto ndop = hlfir::DeclareOp::create(
              opBuilder,
              dop.getLoc(),           
              convertToStaticShape(dop.getResult(0).getType(), staticShape),
              convertToStaticShape(dop.getResult(1).getType(), staticShape), 
              dop.getMemref(),        
              dop.getShape(),         
              dop.getTypeparams(),    
              dop.getDummyScope(),    
              dop.getStorage(),       
              dop.getStorageOffsetAttr(), 
              dop.getUniqNameAttr(),      
              dop.getFortranAttrsAttr(),  
              dop.getDataAttrAttr(),      
              nullptr
              // dop.getDummyArgNoAttr()     
          );
            // auto ndop = hlfir::DeclareOp::create(opBuilder, funcOp.getLoc(), newBoxType, dop.getMemref(), dop.getShape(), dop.getTypeparams(), dop.getDummyScope(), dop.getStorage(), dop.getStorageOffsetAttr(), dop.getUniqNameAttr(), dop.getFortranAttrsAttr(), dop.getDataAttrAttr(), dop.getDummyArgNoAttr());
          dop.replaceAllUsesWith(ndop.getResults());
          dop.erase();
        }
      })
      .Case<hlfir::ElementalOp>([&](hlfir::ElementalOp eop){
        // Example: %6 = hlfir.elemental %0 unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf64> {
        // static ElementalOp create(::mlir::OpBuilder &builder, ::mlir::Location location, mlir::Type result_type, mlir::Value shape, mlir::Value mold = {}, mlir::ValueRange typeparams = {}, bool isUnordered = false);
        if (isDynamicShape(eop.getResult().getType()) && !isDynamicShape(eop.getShape().getType())) {
          assert(shapeMap.contains(eop.getShape()) && "The shape of the elementalOp should be known");
          auto oldResType = eop.getResult().getType();
          auto newResType = convertToStaticShape(oldResType, shapeMap.at(eop.getShape()));
          shapeMap.insert(std::pair(eop.getResult(), shapeMap.at(eop.getShape())));
          opBuilder.setInsertionPoint(eop);
          auto neop = hlfir::ElementalOp::create(
            opBuilder,
            funcOp.getLoc(),
            newResType,
            eop.getShape(),
            eop.getMold(),
            eop.getTypeparams(),
            eop.isOrdered()
          );
          neop.getRegion().takeBody(eop.getRegion());
          eop.replaceAllUsesWith(neop.getResult()); 
          eop.erase();
        }
      })
      .Case<func::FuncOp>([&](func::FuncOp fop){
        auto funcType = funcOp.getFunctionType();
        auto inputTypes = llvm::to_vector(funcType.getInputs());
        auto oldRes = funcType.getResults();

        Block &entryBlock = fop.front();
        for (unsigned i = 0; i < entryBlock.getNumArguments(); i++) {
          auto arg = entryBlock.getArgument(i);
          if (isDynamicShape(arg.getType())) {
            assert(shapeMap.contains(arg) && "Arg Shape should be known!");
            auto staticShape = shapeMap.at(arg); 
            arg.setType(convertToStaticShape(arg.getType(), staticShape));
            inputTypes[i] = convertToStaticShape(inputTypes[i], staticShape);
          }
        }
        auto newFuncType = FunctionType::get(funcOp.getContext(), inputTypes, oldRes);
        funcOp.setType(newFuncType);
      })
      .Default([](auto){});
  }); 
  return;
}

// ShapeInference by tracking constant number
void runShapeInference(MLIRContext& context, mlir::ModuleOp moduleOp, llvm::DenseMap<int, int>& constShapeMap){
  mlir::PassManager pm(&context);
  OpBuilder opBuilder(&context);

  moduleOp->walk([&](func::FuncOp funcOp){
    for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
      if (constShapeMap.contains(i)) {
        valueMap.insert(std::pair<Value, int>(funcOp.getArgument(i), constShapeMap[i]));
      }
    }
    preprocWithExistingPasses(opBuilder, pm, funcOp, constShapeMap);
    shapeInferenceInternal(opBuilder, funcOp);
  });
  return;
}

// Ref: https://openxla.org/xla/aliasing
// XLA code: `xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc`, 
// function: `ConvertToHloModule::RunOnFunction`
void optimizeSignatureForXLAAliasing(MLIRContext* context, func::FuncOp& funcOp) {
  OpBuilder opBuilder(context);
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    funcOp.setArgAttr(i, "tf.aliasing_output", opBuilder.getI64IntegerAttr(i));
  }
  return;
}

// TODO: 
// - Should use target ptrs instead of host, but host has more info, should be changed to use target ptrs later
// - Suppose ArgSizes 4 is shape constant
void getShapeConstantMap(llvm::DenseMap<int, int>& shapeConstMap, int64_t NumHostArgs, void** ArgBasePtrs, int64_t* ArgSizes, int64_t* ArgTypes) {
  for (unsigned i = 0; i < NumHostArgs; i++) {
    auto ty = ArgTypes[i];
    if (isLiteralTy(ty)) {
      int constVal = (int)reinterpret_cast<std::uintptr_t>(ArgBasePtrs[i]);
      shapeConstMap.insert(std::pair(i, constVal));
    };
  }    
  return;
}

// Some arguments are there just meant to be shape meta data, need to drop them for better performance.
// Return a map mapping original Index -> new Index;
llvm::DenseMap<unsigned, unsigned> trimShapeArgs(MLIRContext* context, func::FuncOp& funcOp, int64_t* ArgTypes) {
  llvm::DenseSet<Value> nonShapeArgs;
  funcOp.walk([&](Operation* op){
    if (llvm::isa<func::FuncOp>(op) || llvm::isa<func::ReturnOp>(op)){
      // SKIP
    } else {
      auto operands = op->getOperands();
      std::for_each(operands.begin(), operands.end(), [&](Value operand){
        if (!nonShapeArgs.contains(operand)) {
          nonShapeArgs.insert(operand);
        }
      });
    };
  });

  llvm::DenseSet<unsigned> toKeepIndices;
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    if (!isLiteralTy(ArgTypes[i]) || nonShapeArgs.contains(funcOp.getArgument(i))) {
      toKeepIndices.insert(i);
    };
  }

  // Revise the signature and return value
  llvm::DenseMap<unsigned, unsigned> mappingTable;
  OpBuilder opBuilder(context);
  funcOp.walk([&](Operation * op){
    if (llvm::isa<func::FuncOp>(op)){
      auto funcType = funcOp.getFunctionType();
      auto oldArgsTypes = llvm::to_vector(funcType.getInputs());
      llvm::SmallVector<Type> newArgsTypes;

      unsigned j = 0;
      for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
        if (toKeepIndices.contains(i)) {
          newArgsTypes.push_back(oldArgsTypes[i]); 
          mappingTable[i] = j;
          j += 1;
        }      
      }

      Block &entryBlock = funcOp.front();
      for (int i = entryBlock.getNumArguments() - 1; i >= 0; --i) {
        if (!mappingTable.contains(i)) {
          entryBlock.eraseArgument(i);
        }
      }
      auto newFuncType = FunctionType::get(funcOp.getContext(), newArgsTypes, newArgsTypes);
      funcOp.setType(newFuncType);
    } else if (auto retOp = llvm::dyn_cast<func::ReturnOp>(op)) {
      opBuilder.setInsertionPoint(retOp);

      llvm::SmallVector<Value> retOperands; 
      for (unsigned i = 0; i < retOp.getNumOperands(); i++) {
        if (toKeepIndices.contains(i)) {
          retOperands.push_back(retOp.getOperand(i));
        }
      }
      func::ReturnOp::create(opBuilder, funcOp.getLoc(), retOperands);
      retOp.erase();   
    } else {
      // DO NOTHING
    };
  });
  
  return mappingTable;  
}

