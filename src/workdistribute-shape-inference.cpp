#include "mlir/IR/BuiltinTypes.h"
#include "profiler.h"
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
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;


/// Fill some known values to the mlir and use existing passes to do constant propagation.
/// Including:
/// - CSE: Common Subexpression Elimination
/// - Canonlicalize
/// - SCCP: Sparse Conditional Constant Propagation
/// Ref: https://mlir.llvm.org/docs/Passes/
static void preprocWithExistingPasses(
  OpBuilder opBuilder, 
  PassManager& pm, 
  func::FuncOp funcOp,
  const llvm::DenseMap<Value, int>& valueMap
) {
  auto getSolidVal = [&](Value v) -> std::pair<int, bool> {
    auto it = valueMap.find(v);
    if (it != valueMap.end()) {
      return std::pair(it->getSecond(), true);
    }
    return std::pair(-1, false);
  };

  // Replace some known values with constant values, then lifiting the propagation task to existing mlir passes.
  // llvm::dbgs() << "\n ## before replace known values\n";
  // llvm::dbgs() << "curr funcOP: " << getMLIROperationAsString(funcOp) << "\n";

  llvm::SmallVector<Operation*> opsToDelete;
  funcOp.walk([&](fir::LoadOp lop){

    // llvm::dbgs() <<  "\n on lop: ";
    // lop.print(llvm::dbgs());
    // llvm::dbgs() <<  "\n";
    
    opBuilder.setInsertionPoint(lop);
    auto lopVal = getSolidVal(lop.getOperand());
    if (lopVal.second) {
      auto resValue = lop.getResult();
      auto resType = resValue.getType();
      if (!llvm::isa<mlir::IntegerType>(resType) && !llvm::isa<mlir::IndexType>(resType)) {
        return;
      }

      arith::ConstantIntOp cop = arith::ConstantIntOp::create(opBuilder, funcOp.getLoc(), resValue.getType(), lopVal.first);
      lop.replaceAllUsesWith(cop.getResult());
      assert(lop.use_empty() && "Still been used!");
      opsToDelete.push_back(lop);
    }
  });

  for (auto* op: opsToDelete) {
    op->erase();
  }

  // llvm::dbgs() << "\n ## after replace known values\n";

  // Run passes
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  auto prevSnapshot = getMLIROperationAsString(funcOp);
  auto currSnapshot = std::string();
  const auto MAX_ITERATION = 3;
  auto runPassCount = 0;
  while (true) {
    auto res = pm.run(funcOp); 
    if (mlir::failed(res)) {
      std::cerr << "[Fail] Fail to run passes on funcOp!" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      runPassCount += 1;
    }
    currSnapshot = getMLIROperationAsString(funcOp);
    if ((currSnapshot != prevSnapshot) && (runPassCount < MAX_ITERATION)) {
      std::swap(prevSnapshot, currSnapshot);
      // now prevSnapshot pointing to currSnapshot, currSnapshot will be shadowed in next run.
      continue;
    } else {
      // Is not changed or reach the upper limit
      break;
    }
  }
  return;
}

// Not a roboust transformation but works for now.
static void shapeInferenceInternal(OpBuilder opBuilder, func::FuncOp funcOp, llvm::DenseMap<Value, llvm::SmallVector<int>>& sliceShiftMap) {
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
        for (int i = 0; i < sop.getNumOperands(); i++) {
          auto opr = sop.getOperand(i);
          assert(constTrackingMap.contains(opr) && "Operand static value should be known!");
          sizes.push_back(constTrackingMap.at(opr));
        }
        shapeMap.insert(std::pair(sop.getResult(), sizes)); // %0 -> {1000, 1000}
      })
      .Case<fir::ShapeShiftOp>([&](fir::ShapeShiftOp ssOp){
        // In Fortran, the idx can start from any number, often we have shapeShiftOp
        // %10 = fir.shape_shift %c-1, %c964, %c-1, %c965 : (index, index, index, index) -> !fir.shapeshift<2> 
        // The shape is rank=2:
        // - first rank: lower bound: %c-1, len: %c964
        // - second rank: lower bound: %c-1, len: %c965
        //
        // TODO: the lower bound should also be stored for later usage: for example, when translating from `hlfir::desinateOp` to `stablehlo::slicingOp`
        llvm::SmallVector<int64_t> shapeSizes;
        llvm::SmallVector<int> shapeShifts;
        assert(ssOp.getNumOperands()%2 == 0 && "ShapeShift should have even number of operands!");
        shapeSizes.reserve(ssOp.getNumOperands()/2);

        for (int i = 0; i < ssOp.getNumOperands(); i++) {
          auto opr = ssOp.getOperand(i);
          assert(constTrackingMap.contains(opr) && "Operand static value of shapeShift should be known!");
          if (i % 2 == 0) {
            auto idxLowerBound = constTrackingMap.at(opr);
            shapeShifts.push_back(idxLowerBound);
          } else {
            auto dimSize = constTrackingMap.at(opr);
            shapeSizes.push_back(dimSize);
          }
        }
        shapeMap.insert(std::pair(ssOp.getResult(), shapeSizes)); // %10 -> {%c964, %c965}
        sliceShiftMap.insert(std::pair(ssOp.getResult(), shapeShifts)); // %10 -> {%c-1, %c-1}
      })
      .Case<hlfir::DeclareOp>([&](hlfir::DeclareOp dop){
        // Example: %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFcoexecute_aEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
        // Objective: %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFcoexecute_aEz"} : (!fir.ref<!fir.array<1000x1000xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<1000x1000xf64>>, !fir.ref<!fir.array<1000x1000xf64>>)
        if (isDynamicShape(dop.getResult(0).getType())) {
          assert(shapeMap.contains(dop.getShape()) && "The shape of the declareOp has not been added!!!!");
          auto staticShape = shapeMap.at(dop.getShape());

          // propagate the shape of the results
          for (int i = 0; i < dop.getNumResults(); i++) {
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
      .Case<hlfir::DesignateOp>([&](hlfir::DesignateOp dop){
        // %5 = hlfir.designate %3#0 (%c1:%c5:%c1)  shape %4 : (!fir.box<!fir.array<?xf64>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
        // In the above example, it creates a part-ref of %3#0, with starting index %c1, end index %c5 and stride %c1
        // in such case, the shape will be different
        // But it can also be something a simple form, referring just a single value of it:
        // example: %451 = "hlfir.designate"(%447#0, %arg9) 
        if (isDynamicShape(dop.getResult().getType())) {
          auto shapeVal = dop.getShape();
          auto it = shapeMap.find(shapeVal);
          assert(it != shapeMap.end());
          auto shapeInfo = it->getSecond();
          opBuilder.setInsertionPoint(dop);
          auto neoDop = hlfir::DesignateOp::create(
            opBuilder,
            dop.getLoc(),
            convertToStaticShape(dop.getResult().getType(), shapeInfo),
            dop.getMemref(),
            dop.getComponentAttr(),
            dop.getComponentShape(),
            dop.getIndices(),          
            dop.getIsTripletAttr(),    
            dop.getSubstring(),
            dop.getComplexPartAttr(),
            dop.getShape(),           
            dop.getTypeparams(),
            dop.getFortranAttrsAttr()
          );
          dop.replaceAllUsesWith(neoDop.getResult());
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
        for (int i = 0; i < entryBlock.getNumArguments(); i++) {
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

void inferShape(
  MLIRContext* ctx,
  ModuleOp moduleOp,
  int64_t NumHostArgs, 
  void** ArgBasePtrs, 
  int64_t* ArgSizes, 
  int64_t* ArgTypes,
  llvm::DenseMap<Value, llvm::SmallVector<int>>& sliceShiftMap 
) {
  PROFILE_SCOPE("shape infer", Phase::LOWERING_SHAPE_INFER)
  // Key: index of the arguments of the function, value: if the argment is literal type, we know the value of the arg 
  llvm::DenseMap<uint, uint> shapeConstMap;
  llvm::DenseMap<Value, int> valueMap;

  mlir::PassManager pm(ctx);
  OpBuilder opBuilder(ctx);

  // some parameters containing the shape info are passed as pointer like
  moduleOp.walk([&](func::FuncOp funcOp){

    assert(NumHostArgs == funcOp.getNumArguments() && "NumHostArgs is not equal to funcOp args count!!");
    for (int i = 0; i < funcOp.getNumArguments(); i++) {
      auto ty = ArgTypes[i];
      if (isLiteralTy(ty)) {
        int constVal = (int)reinterpret_cast<std::uintptr_t>(ArgBasePtrs[i]);
        valueMap.insert(std::pair<Value, int>(funcOp.getArgument(i), constVal));
      }
    };
  });
  // llvm::dbgs() << "\n # after get arguments\n";

  moduleOp->walk([&](hlfir::DeclareOp dop){
    // Sometimes it's included in declare Op
    // %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFcoexecute_aEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    // ...
    // %4 = fir.load %2#0 : !fir.ref<i32>
    if (dop.getNumOperands() == 1 && valueMap.contains(dop.getOperand(0)) && dop.getNumResults() > 0) {
      valueMap.insert(std::pair<Value, int>(dop.getResults()[0], valueMap.lookup(dop.getOperand(0))));
    } 
  }); 
  
  // llvm::dbgs() << "\n # after walk dops\n";

  for (auto funcOp: moduleOp.getOps<func::FuncOp>()) {
    for (int i = 0; i < funcOp.getNumArguments(); i++) {
      if (shapeConstMap.contains(i)) {
        valueMap.insert(std::pair<Value, int>(funcOp.getArgument(i), shapeConstMap.at(i)));
      }
    }

    // std::cerr << "\nBefore Prepro: ====================================\n";
    preprocWithExistingPasses(opBuilder, pm, funcOp, valueMap);
    // llvm::dbgs() << "\n ## after preproc with exesiting passes\n";

    // std::cerr << "\nAfter Prepro: \n" << getMLIROperationAsString(funcOp);

    shapeInferenceInternal(opBuilder, funcOp, sliceShiftMap);
    // llvm::dbgs() << "\n ## after shape inference internal\n";
  }
  return;
}

