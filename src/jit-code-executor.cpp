#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockSupport.h>
#include "mlir/IR/Builders.h"
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <omp.h>
#include <ostream>
#include <sys/types.h>
#include "kernel_launcher.h"
#include "kernel_pointer_interface.h"

using namespace mlir;


// void getShapeConstantMap(llvm::DenseMap<uint, uint>& shapeConstMap, int64_t NumHostArgs, void** ArgBasePtrs, int64_t* ArgSizes, int64_t* ArgTypes);
// void runShapeInference(MLIRContext* context, mlir::ModuleOp moduleOp, llvm::DenseMap<uint, uint>& constShapeMap);

void inferShape(MLIRContext* ctx, ModuleOp moduleOp, int64_t NumHostArgs, void** ArgBasePtrs, int64_t* ArgSizes, int64_t* ArgTypes); 

void optimizeSignatureForXLAAliasing(MLIRContext* context, func::FuncOp& funcOp);

func::FuncOp workdistributeToStableHLO(MLIRContext* context, const mlir::ModuleOp& moduleOp);

llvm::DenseMap<unsigned, unsigned> trimShapeArgs(MLIRContext* context, func::FuncOp& funcOp, int64_t* ArgTypes);


// ------------------------------ Init ------------------------------ 
/**
  * JitCode: A function contains the omp::TargetOp with a omp::workdistributeOp inside.
  *
  */
extern "C" int64_t __botw_jit_code(void *JitCode, int64_t NumArgs,
                                   void **TgtArgs, ptrdiff_t *TgtOffsets,
                                   void *DeviceArgs, int64_t NumHostArgs,
                                   void **ArgBasePtrs, void **ArgPtrs,
                                   int64_t *ArgSizes, int64_t *ArgTypes,
                                   void **ArgNames) {
  char *JitCodeC = reinterpret_cast<char *>(JitCode);
  std::cerr << "Got a jit call with " << NumArgs << " args into:\n" << JitCodeC << "\n";
  
  // Parse JitCode to ModuleOp
  MLIRContext* ctx = JitManager::getInstance().getContext();
  auto JitCodePtrUint = reinterpret_cast<uintptr_t>(JitCode);
  ModuleOp moduleOp = JitManager::getInstance().getModuleOp(JitCodePtrUint, JitCodeC);

  inferShape(ctx, moduleOp, NumHostArgs, ArgBasePtrs, ArgSizes, ArgTypes);

  std::cerr << "\nAfter Shape Infer: \n" << getMLIROperationAsString(moduleOp) << "\n";

  func::FuncOp kernelFunc = workdistributeToStableHLO(ctx, moduleOp);

  optimizeSignatureForXLAAliasing(ctx, kernelFunc);

  llvm::DenseMap<unsigned, unsigned> argsIndicesMapping = trimShapeArgs(ctx, kernelFunc, ArgTypes);
   std::cerr << "Transform the jit call into:\n" << getMLIROperationAsString(kernelFunc) << "\n";
  

  // ------------------------------ Fill the kernel args ------------------------------ 
  auto argTypes = kernelFunc.getFunctionType().getInputs(); 
  TensorDesc newArgs[kernelFunc.getNumArguments()];  // args after being trimmed

  for (int oldIdx = 0; oldIdx < NumHostArgs && argsIndicesMapping.contains(oldIdx); oldIdx++){
    auto newIdx = argsIndicesMapping.at(oldIdx);
    auto thisTy = argTypes[newIdx]; 
    assert(llvm::isa<RankedTensorType>(thisTy) && "Suppose all args are ");
    auto rtType = llvm::dyn_cast<RankedTensorType>(thisTy);

    newArgs[newIdx] = (struct TensorDesc){
      .data = TgtArgs[oldIdx],
      .shape = rtType.getShape().data(),
      .rank = (int32_t)rtType.getRank(),
      .dtype = [&](){
        auto eleType = rtType.getElementType();
        if (eleType.isF32()){
          return DType::F32;
        } else if (eleType.isF64()){
          return DType::F64;
        } else if (eleType.isInteger(32)) {
          return DType::I32;
        } else if (eleType.isInteger(64)) {
          return DType::I64;
        } else {
          llvm::errs() << "\n";
          eleType.print(llvm::errs() << "Unknown input type: ");
          llvm::errs() << "\n";
          exit(EXIT_FAILURE);
        }
      }(),
      .isLiteral = isLiteralTy(ArgTypes[oldIdx]),
    };
  }

  KernelArgs args = (struct KernelArgs){
    .inputArgCount = argsIndicesMapping.size(),
    .inputArgs = newArgs,
    .outputArgCount = argsIndicesMapping.size(),
    .outputArgs = newArgs,
    .targetDevice = TargetDevice::CPU,
  };


  // ------------------------------ Fill the kernel args ------------------------------ 
  launchKernel(&args, JitCodePtrUint, getMLIROperationAsString(kernelFunc));
  return 0;
}
