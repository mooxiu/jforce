#include "../support/kernel_pointer_interface.h"
#include "../support/profiler.h"
#include "../support/utilities.h"
#include "../transform/passes.h"
#include "jit-manager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace mlir;

#define JIT_LITERAL_VAL_ATTR_NAME "jit.literal_val"
#define JIT_ARGS_MAPPING_ATTR_NAME "jit.args_mapping"


void inferShape(MLIRContext *ctx, ModuleOp moduleOp, int64_t NumHostArgs,
                void **ArgBasePtrs, int64_t *ArgSizes, int64_t *ArgTypes,
                llvm::DenseMap<Value, llvm::SmallVector<int>> &sliceShiftMap);

void optimizeSignatureForXLAAliasing(MLIRContext *context,
                                     func::FuncOp &funcOp);

func::FuncOp workdistributeToStableHLO(
    MLIRContext *context, const mlir::ModuleOp &moduleOp,
    const llvm::DenseMap<Value, llvm::SmallVector<int>> &sliceShiftMap);

llvm::DenseMap<unsigned, unsigned>
trimShapeArgs(MLIRContext *context, func::FuncOp &funcOp, int64_t *ArgTypes);

static TargetDevice getTargetDevice() {
#ifdef TARGET_DEVICE
  if constexpr (std::string_view(TARGET_DEVICE) == "CUDA") {
    return TargetDevice::CUDA;
  } else if constexpr (std::string_view(TARGET_DEVICE) == "ROCM") {
    return TargetDevice::ROCM;
  } else if constexpr (std::string_view(TARGET_DEVICE) == "TPU") {
    return TargetDevice::TPU;
  }
  return TargetDevice::CPU;
#else
  return TargetDevice::CPU;
#endif
}

bool fillKernelFuncArgs(llvm::DenseMap<unsigned, unsigned> argsIndicesMapping,
                        TensorDesc *newArgs, ArrayRef<Type> kernelFuncTypes,
                        int64_t NumHostArgs, int64_t *ArgTypes,
                        void **TgtArgs) {
  for (int oldIdx = 0; oldIdx < NumHostArgs; oldIdx++) {
    if (!argsIndicesMapping.contains(oldIdx)) {
      continue;
    }
    auto newIdx = argsIndicesMapping.at(oldIdx);
    auto thisTy = kernelFuncTypes[newIdx];
    assert(llvm::isa<RankedTensorType>(thisTy) && "Suppose all args are ");
    auto rtType = llvm::dyn_cast<RankedTensorType>(thisTy);

    newArgs[newIdx] = TensorDesc{
        .data = TgtArgs[oldIdx],
        .shape = rtType.getShape().data(),
        .rank = (int32_t)rtType.getRank(),
        .dtype = getDTypeFromRankedTensorType(rtType),
        .isLiteral = isLiteralTy(ArgTypes[oldIdx]),
    };
  }
  return true;
}

struct jitArg {
  void *hostPtr;
  void *tgtPtr;
  int64_t size;
  bool isLiteral;
};

llvm::SmallVector<jitArg> packJitArg(int64_t NumArgs, void **TgtArgs,
                                     void **ArgPtrs, int64_t *ArgSizes,
                                     int64_t *ArgTypes) {
  llvm::SmallVector<jitArg> args;
  args.resize(NumArgs);

  for (int i = 0; i < NumArgs; i++) {
    args[i] = jitArg{
      .hostPtr = ArgPtrs[i],
      .tgtPtr = TgtArgs[i],
      .size = ArgSizes[i],
      .isLiteral = isLiteralTy(ArgTypes[i]),
    };
  };
  return args;
}

void insertJitInfo(mlir::OpBuilder& builder, func::FuncOp kernelFunc, llvm::SmallVector<jitArg> args) {
  llvm::SmallVector<mlir::Attribute> argsAttr;
  for (int i = 0; i < args.size(); i++) {
    llvm::SmallVector<mlir::NamedAttribute> perArgAttr;

    perArgAttr.push_back(builder.getNamedAttr("jit.arg_size", builder.getI64IntegerAttr(args[i].size)));
    if (args[i].isLiteral) {
      perArgAttr.push_back(builder.getNamedAttr(JIT_LITERAL_VAL_ATTR_NAME, builder.getUnitAttr()));
    }
    argsAttr.push_back(builder.getDictionaryAttr(perArgAttr));
  }

  kernelFunc.setArgAttrsAttr(builder.getArrayAttr(argsAttr));
  return;
}

// ------------------------------ Init ------------------------------
/**
 * JitCode: A function contains the omp::TargetOp with a omp::workdistributeOp
 * inside.
 *
 */
extern "C" int64_t __botw_jit_code(void *JitCode, int64_t NumArgs,
                                   void **TgtArgs, ptrdiff_t *TgtOffsets,
                                   void *DeviceArgs, int64_t NumHostArgs,
                                   void **ArgBasePtrs, void **ArgPtrs,
                                   int64_t *ArgSizes, int64_t *ArgTypes,
                                   void **ArgNames) {
  PROFILE_SCOPE("total", Phase::TOTAL);
  char *JitCodeC = reinterpret_cast<char *>(JitCode);
  // std::cerr << "Got a jit call with " << NumArgs << " args into:\n" <<
  // JitCodeC << "\n"; llvm::dbgs() << "\nreceive a jit call\n";

  // #define p(A) std::cerr << " " << #A << ": " << A[I] << "\n"
  // #define h(A) \
  //   std::cerr << " " << #A << std::hex << ": 0x" << A[I] << std::dec << "\n"
  //   for (unsigned I = 0; I < NumArgs; I++) {
  //     std::cerr << "Device Arg #" << I << ":\n";
  //     p(TgtArgs);
  //     p(TgtOffsets);
  //   }
  //   for (unsigned I = 0; I < NumHostArgs; I++) {
  //     std::cerr << "Host Arg #" << I << ":\n";
  //     p(ArgBasePtrs);
  //     p(ArgPtrs);
  //     p(ArgSizes);
  //     h(ArgTypes);
  //     h(ArgNames);
  //   }
  // #undef p
  // #undef h

  assert(NumArgs == NumHostArgs);

  auto JitCodePtrUint = reinterpret_cast<uintptr_t>(JitCode);
  auto l1JitMetas = JitManager::getInstance().tryGetL1JitMetas(JitCodePtrUint);
  llvm::SmallVector<uint64_t, 128> l2Key;

  if (l1JitMetas != nullptr) {
    l2Key = JitManager::getInstance().getL2JitMetasKey(
        NumArgs, ArgTypes, TgtArgs, ArgSizes, JitCodePtrUint,
        l1JitMetas->argsIndices);
    auto l2JitMetas = JitManager::getInstance().tryGetL2JitMetas(l2Key);

    if (l2JitMetas != nullptr) {
      // auto kernelFunc = jitMeta->kernelFunc;
      std::vector<TensorDesc> newArgs(l2JitMetas->kernelFuncTypes.size());
      fillKernelFuncArgs(l2JitMetas->argsIndicesMapping, newArgs.data(),
                         l2JitMetas->kernelFuncTypes, NumHostArgs, ArgTypes,
                         TgtArgs);
      auto kArgs =
          (KernelArgs){.inputArgCount = l2JitMetas->argsIndicesMapping.size(),
                       .inputArgs = newArgs.data(),
                       .outputArgCount = l2JitMetas->argsIndicesMapping.size(),
                       .outputArgs = newArgs.data(),
                       .targetDevice = getTargetDevice()};
      JitManager::getInstance().launchKernel(
          l2JitMetas->exe, &kArgs, JitCodePtrUint, l2JitMetas->kernelFuncStr);
      return 0;
    }
  }

  // Parse JitCode to ModuleOp
  MLIRContext *ctx = JitManager::getInstance().getContext();
  // Use OweningOpRef so RAII can help to destroy the tree
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp = JitManager::getInstance().getModuleOp(JitCodePtrUint, JitCodeC);
  mlir::OpBuilder builder(ctx);
  auto args = packJitArg(NumArgs, TgtArgs, ArgPtrs, ArgSizes, ArgTypes);
  auto kernel = moduleOp.get().lookupSymbol<func::FuncOp>("kernel");
  assert(kernel && "FuncOp with name kernel should exist!");
  insertJitInfo(builder, kernel, args);

  mlir::PassManager pm(ctx);
  pm.addPass(xla_jit::createShapeInferPass());
  pm.addPass(xla_jit::createWorkdistributeToStableHLOPass());
  pm.addPass(xla_jit::createAliasingPass());
  pm.addPass(xla_jit::createTrimArgsPass());

  if (mlir::failed(pm.run(moduleOp.get()))) {
    llvm::errs() << "MLIR Pass Pipeline failed!\n";
    std::exit(EXIT_FAILURE);
  }

  
  // TODO: gradually change the functions into standard passes compatible with MLIR ecosystem!!!!
  DEBUG_PRINT("\nThe module we got: \n" +
              getMLIROperationAsString(moduleOp.get()));

  llvm::DenseMap<Value, llvm::SmallVector<int>> sliceShiftMap;
  inferShape(ctx, moduleOp.get(), NumHostArgs, ArgBasePtrs, ArgSizes, ArgTypes,
             sliceShiftMap);
  DEBUG_PRINT("\nAfter shape Infer:\n" +
              getMLIROperationAsString(moduleOp.get()));

  func::FuncOp kernelFunc =
      workdistributeToStableHLO(ctx, moduleOp.get(), sliceShiftMap);
  DEBUG_PRINT("\nAfter lowering to wd:\n" +
              getMLIROperationAsString(kernelFunc));

  optimizeSignatureForXLAAliasing(ctx, kernelFunc);

  llvm::DenseMap<unsigned, unsigned> argsIndicesMapping =
      trimShapeArgs(ctx, kernelFunc, ArgTypes);
  DEBUG_PRINT("\nAfter trim shape args:\n" +
              getMLIROperationAsString(kernelFunc));

  if (l1JitMetas == nullptr) {
    llvm::DenseSet<int> argsIndices;
    for (int i = 0; i < NumArgs; i++) {
      // not contains in argsIndicesMapping, meaning it's the shape arguments
      // that been trimmed above
      if (!argsIndicesMapping.contains(i)) {
        argsIndices.insert(i);
      }
    }
    l2Key = JitManager::getInstance().getL2JitMetasKey(
        NumArgs, ArgTypes, TgtArgs, ArgSizes, JitCodePtrUint, argsIndices);
    JitManager::getInstance().saveL1JitMetas(JitCodePtrUint,
                                             std::move(argsIndices));
  }

  auto createdL2JitMetas = JitManager::getInstance().createL2JitMetas(
      l2Key, kernelFunc, argsIndicesMapping, getTargetDevice());

  std::vector<TensorDesc> newArgs(createdL2JitMetas->kernelFuncTypes.size());

  if (fillKernelFuncArgs(argsIndicesMapping, newArgs.data(),
                         createdL2JitMetas->kernelFuncTypes, NumHostArgs,
                         ArgTypes, TgtArgs)) {
    KernelArgs args = (struct KernelArgs){
        .inputArgCount = argsIndicesMapping.size(),
        .inputArgs = newArgs.data(),
        .outputArgCount = argsIndicesMapping.size(),
        .outputArgs = newArgs.data(),
        .targetDevice = getTargetDevice(),
    };
    JitManager::getInstance().launchKernel(createdL2JitMetas->exe, &args,
                                           JitCodePtrUint,
                                           createdL2JitMetas->kernelFuncStr);
  } else {
    std::cerr << "Fail to fill kernel func args!\n";
    return 1;
  }

  return 0;
}
