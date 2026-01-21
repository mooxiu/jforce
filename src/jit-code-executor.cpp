#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "kernel_pointer_interface.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
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
#include <ostream>

using namespace mlir;

func::FuncOp workdistributeToStableHLO(MLIRContext& context, const mlir::ModuleOp& moduleOp);

void launch_kernel(KernelArgs *argsPointer, const std::string& kernelFuncStr);

void runShapeInference(MLIRContext& context, mlir::ModuleOp moduleOp, llvm::DenseMap<int, int>& constShapeMap);


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

// TODO: 
// - Should use target ptrs instead of host, but host has more info, should be changed to use target ptrs later
// - Suppose ArgSizes 4 is shape constant
static void getShapeConstantMap(llvm::DenseMap<int, int>& shapeConstMap, int64_t NumHostArgs, void** ArgBasePtrs, int64_t* ArgSizes, int64_t* ArgTypes) {
  // Is Immediate value, the pointer addr is the value of the constant
  auto isImm = [](int64_t ty) -> bool {
    // 0x100 means `mapping is literal`
    // ref: `offload/include/omptarget.h`
    // TODO: include header file instead of using the number directly
    return ty&0x100;
  };
  
  for (unsigned i = 0; i < NumHostArgs; i++) {
    auto ty = ArgTypes[i];
    if (isImm(ty)) {
      int constVal = (int)reinterpret_cast<std::uintptr_t>(ArgBasePtrs[i]);
      shapeConstMap.insert(std::pair(i, constVal));
    };
  }    
  return;
}

// Some arguments are there just meant to be shape meta data, need to drop them for better performance.
// Return a map mapping original Index -> new Index;
static llvm::DenseMap<unsigned, unsigned> trimShapeArgs(MLIRContext* context, func::FuncOp& funcOp, int64_t* ArgTypes) {
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
    if ((ArgTypes[i]&0x100) == 0 || nonShapeArgs.contains(funcOp.getArgument(i))) {
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
  std::cout << "JIT Code: \n" << JitCodeC << std::endl;
  auto moduleOp = module.get();


  llvm::DenseMap<int, int> constShapeMap; // key: arg index; value: integer literal value 
  getShapeConstantMap(constShapeMap, NumHostArgs, ArgBasePtrs, ArgSizes, ArgTypes);
  runShapeInference(context, moduleOp, constShapeMap);
  func::FuncOp kernelFunc = workdistributeToStableHLO(context, moduleOp);
  optimizeSignatureForXLAAliasing(&context, kernelFunc);
  llvm::DenseMap<unsigned, unsigned> mappingTable = trimShapeArgs(&context, kernelFunc, ArgTypes);
  std::string kernelFuncLiteral = getFuncOpAsString(kernelFunc);
  std::cout << "Function lowered from JIT Code: \n" << kernelFuncLiteral << std::endl;


  //********************Execution********************
  // Fill the kernel args
  KernelArgs args;
  args.targetDevice = TargetDevice::CUDA;
  auto argTypes = kernelFunc.getFunctionType().getInputs(); 
  TensorDesc inputArgs[kernelFunc.getNumArguments()]; // input arguments should be all args
  TensorDesc outputArgs[kernelFunc.getNumArguments()];

  // TODO: using NumHostArgs may not be very robostic, here i means the index in the function arguments beform trimming.
  for (unsigned i = 0; i < NumHostArgs && mappingTable.contains(i); i++){
    auto newIdx = mappingTable.at(i);
    auto thisTy = argTypes[newIdx]; 
    assert(llvm::isa<RankedTensorType>(thisTy) && "Suppose all args are ");
    auto rtType = llvm::dyn_cast<RankedTensorType>(thisTy);

    inputArgs[newIdx].data = TgtArgs[i];
    inputArgs[newIdx].shape = rtType.getShape().data();
    inputArgs[newIdx].rank = rtType.getRank();
    inputArgs[newIdx].dtype = [&](){
      auto eleType = rtType.getElementType();
      if (eleType.isF32()){
        return DType::F32;
      } else if (eleType.isF64()){
        return DType::F64;
      } else if (eleType.isInteger(32)) {
        return DType::I32;
      }else {
        std::cerr << "Unknown input type!\n";
        exit(EXIT_FAILURE);
      }
    }();
    inputArgs[newIdx].isLiteral = (ArgTypes[i] & 0x100);
  }

  args.inputArgs = inputArgs;
  args.inputArgCount = mappingTable.size();
  args.outputArgs = inputArgs; 
  args.outputArgCount = mappingTable.size();

  args.formatPrint();



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


  launch_kernel(&args, kernelFuncLiteral);

  return 0;
}
