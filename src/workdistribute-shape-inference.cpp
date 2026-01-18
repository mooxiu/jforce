#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <iostream>
#include <mlir/Support/LLVM.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlir;

static bool withUnknownShape(Value v) {
  return true;
}

// ShapeInference by tracking constant number
void shapeInference(mlir::ModuleOp& moduleOp, 
                    void *JitCode, int64_t NumArgs,
                    void **TgtArgs, ptrdiff_t *TgtOffsets,
                    void *DeviceArgs, int64_t NumHostArgs,
                    void **ArgBasePtrs, void **ArgPtrs,
                    int64_t *ArgSizes, int64_t *ArgTypes,
                    void **ArgNames){


  if (!llvm::isa<func::FuncOp>(moduleOp)) {
    std::cerr << "Expect ModuleOp To Be A Function!" << std::endl;
  }
  func::FuncOp targetFunc = llvm::dyn_cast<func::FuncOp>(moduleOp);

  mlir::DenseSet<Value> shapeUnknownArgs;
  mlir::DenseMap<Value, int64_t> scalarMap; // This only contains meta info relative to shapes
  mlir::DenseSet<Value, Value> trackingMap; // Ideally, scalarMap can track all the shapes, but if not, adding to trackingMap for now.
  mlir::DenseMap<Value, std::vector<int>> shapeMap;

  // Scanning Phase:
  int argsCount = targetFunc.getNumArguments();
  for (int i = 0; i < argsCount; i++) {
    Value arg = targetFunc.getArgument(i);
    if (withUnknownShape(arg)) {
      shapeUnknownArgs.insert(arg);
    }
  }
  
  targetFunc.walk([&](Operation* op) -> void {
    llvm::TypeSwitch<Operation *>(op)
      .Case<arith::ConstantOp>([&](arith::ConstantOp op){
        // example: %c0 = arith.constant 0 : index
        auto attr = op.getValue(); 
        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)){
          scalarMap[op.getResult()] = intAttr.getInt();
        } else {
          llvm::dbgs() << "Unhandled arith::ConstantOp: \n";
          op.print(llvm::dbgs());
        };
      })
      .Case<fir::LoadOp>([&](fir::LoadOp lop){
         //example: %0 = fir.load %arg9 : !fir.ref<i32> 
        
      })
      .Default([](auto){});
  });
  // Rebuilding Phase:

  assert(shapeUnknownArgs.empty() && "Expect no shape unknown vars at the ned of this pass");
  return;
}

