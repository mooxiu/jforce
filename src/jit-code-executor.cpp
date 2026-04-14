#include "kernel_pointer_interface.h"
#include "jit-manager.h"
#include "profiler.h"
#include "utilities.h"
#include <iostream>

using namespace mlir;

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
    };
  }
  return true;
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
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
      JitManager::getInstance().getModuleOp(JitCodePtrUint, JitCodeC);
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
