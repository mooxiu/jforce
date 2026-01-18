#include "../third_party/headers/pjrt_c_api.h"
#include "absl/strings/internal/str_format/extension.h"
#include "flang/Support/Fortran.h"
#include "kernel_pointer_interface.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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
#include <sstream>
#include <utility>
#include <vector>

using namespace mlir;

func::FuncOp workdistributeToStableHLO(MLIRContext& context, const ModuleOp& moduleOp);

void launch_kernel(KernelArgs *argsPointer, const std::string& kernelFuncStr);

// TODO: Adding verifications for the input moduleOp
static bool verifyJitCode(const ModuleOp& moduleOp) {
  return true;
}

static std::string getFuncOpAsString(func::FuncOp funcOp) {
  std::string output;
  llvm::raw_string_ostream os(output);
  funcOp.print(os);
  return output;
}

// Ref: https://openxla.org/xla/aliasing
// XLA code: `xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc`, 
// function: `ConvertToHloModule::RunOnFunction`
static void optimizeSignatureForXLAAliasing(MLIRContext* context, func::FuncOp& funcOp) {
  OpBuilder opBuilder(context);
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    funcOp.setArgAttr(i, "tf.aliasing_output", opBuilder.getI64IntegerAttr(i));
  }
  return;
}


// XLA cannot update input buffers in-place, we have to distinguish input and output
// Return value is a vector of output indices
static std::vector<int> optimizeSignature(MLIRContext* context, func::FuncOp& funcOp) {
  auto& body = funcOp.getFunctionBody();
  auto returnOp = llvm::cast<func::ReturnOp>(body.back().getTerminator());
  // Before optimization, the input args should be the same size of returned
  std::vector<int> outputArgs;
  std::vector<Value> outputValues;
  std::vector<Type> outputTypes;
  funcOp.walk([&](mlir::func::ReturnOp rop){
    for (unsigned i = 0; i < rop.getNumOperands(); i++) {
      if (funcOp.getArgument(i) != rop.getOperand(i)) {
        outputArgs.push_back(i);
        outputValues.push_back(rop.getOperand(i));
        outputTypes.push_back(rop.getOperand(i).getType());
      }
    };
  });
  // update return values
  OpBuilder opBuilder(context);
  opBuilder.setInsertionPoint(returnOp);
  func::ReturnOp::create(opBuilder, returnOp.getLoc(), outputValues);
  returnOp.erase();

  // update the function signature
  auto revisedFuncType = FunctionType::get(context, funcOp.getArgumentTypes(), outputTypes);
  funcOp.setFunctionType(revisedFuncType);

  return outputArgs;
}

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
  // std::cerr << "Got a jit call with " << NumArgs << " args into:\n" << JitCodeC << "\n";
  

  // Parse JitCode to ModuleOp
  mlir::MLIRContext context;
  context.loadDialect<
    func::FuncDialect, 
    omp::OpenMPDialect, 
    fir::FIROpsDialect, 
    hlfir::hlfirDialect, 
    arith::ArithDialect, 
    stablehlo::StablehloDialect>();
  mlir::ParserConfig parserConfig(&context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(JitCodeC, parserConfig);
  if (!module) {
    std::cerr << "Module not extracted!" << std::endl;
    exit(EXIT_FAILURE);
  }
  auto moduleOp = module.get();
  std::cout << "JIT Code: \n" << JitCodeC << std::endl;

  func::FuncOp kernelFunc = workdistributeToStableHLO(context, moduleOp);
  // auto realReturnedIndices = optimizeSignature(&context, kernelFunc);
  optimizeSignatureForXLAAliasing(&context, kernelFunc);
  std::string kernelFuncLiteral = getFuncOpAsString(kernelFunc);
  std::cout << "Function lowered from JIT Code: \n" << kernelFuncLiteral << std::endl;


  //********************Execution********************
  // Fill the kernel args
  KernelArgs args;
  args.targetDevice = TargetDevice::CPU;
  auto argTypes = kernelFunc.getFunctionType().getInputs(); 
  TensorDesc inputArgs[kernelFunc.getNumArguments()]; // input arguments should be all args
  TensorDesc outputArgs[kernelFunc.getNumArguments()];

  assert(kernelFunc.getNumArguments() == NumArgs && "Fail to assert kernelFun Args Count = NumArgs");
  // FIXME: NumArgs may contains constants, which is not in target
  for (unsigned i = 0; i < NumArgs; i++) {
    auto thisTy = argTypes[i]; 
    assert(llvm::isa<RankedTensorType>(thisTy) && "Suppose all args are ");
    auto rtType = llvm::dyn_cast<RankedTensorType>(thisTy);

    inputArgs[i].data = TgtArgs[i];
    inputArgs[i].shape = rtType.getShape().data();
    inputArgs[i].rank = rtType.getRank();
    inputArgs[i].dtype = DType::F32;
  }
  args.inputArgs = inputArgs;
  args.inputArgCount = NumArgs;
  args.outputArgs = inputArgs; 
  args.outputArgCount = NumArgs;

  args.formatPrint();

  launch_kernel(&args, kernelFuncLiteral);
  return 0;


#define p(A) std::cerr << " " << #A << ": " << A[I] << "\n"
#define h(A) std::cerr << " " << #A << std::hex << ": 0x" << A[I] << std::dec << "\n"

  for (unsigned I = 0; I < NumArgs; I++) {
    std::cerr << "Device Arg #" << I << ":\n";
    p(TgtArgs);
    p(TgtOffsets);
  }
  for (unsigned I = 0; I < NumHostArgs; I++) {
    std::cerr << "Host Arg #" << I << ":\n";
    p(ArgBasePtrs);
    p(ArgPtrs);
    p(ArgSizes);
    h(ArgTypes);
  }

#undef p
#undef h

  return 0;
}
