#include "../support/profiler.h"
#include "../support/utilities.h"
#include "jit-manager.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace mlir;

llvm::SmallVector<TensorDesc>
assembleXLAFuncArgs(ArrayRef<Type> kernelFuncTypes, int64_t argCount,
                    int64_t *ArgTypes, void **TgtArgs) {
  llvm::SmallVector<TensorDesc> XLAFuncArgs;
  XLAFuncArgs.resize(argCount);

  for (int i = 0; i < argCount; i++) {
    auto thisTy = kernelFuncTypes[i];
    assert(llvm::isa<RankedTensorType>(thisTy) && "Suppose all args are ");
    auto rtType = llvm::dyn_cast<RankedTensorType>(thisTy);

    XLAFuncArgs[i] = TensorDesc{
        .data = TgtArgs[i],
        .shape = rtType.getShape(),
        .rank = (int32_t)rtType.getRank(),
        .dtype = getDTypeFromRankedTensorType(rtType),
        .isLiteral = isLiteralTy(ArgTypes[i]),
    };
  }
  return XLAFuncArgs;
}

static void checkDeletgatedLaunchInputs(void *JitCode, int64_t NumArgs,
                                        void **TgtArgs, ptrdiff_t *TgtOffsets,
                                        int64_t NumHostArgs, void **ArgBasePtrs,
                                        void **ArgPtrs, int64_t *ArgSizes,
                                        int64_t *ArgTypes, void **ArgNames) {
  char *JitCodeC = reinterpret_cast<char *>(JitCode);
  llvm::dbgs() << "Got a jit call with " << NumArgs << " args into:\n"
               << JitCodeC << "\n";
#define p(A) std::cerr << " " << #A << ": " << A[I] << "\n"
#define h(A)                                                                   \
  std::cerr << " " << #A << std::hex << ": 0x" << A[I] << std::dec << "\n"
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
    h(ArgNames);
  }
#undef p
#undef h
  assert(NumArgs == NumHostArgs);
}

// ------------------------------ Init ------------------------------
//

extern "C" {

__attribute__((visibility("default"))) void RetrieveData(void *hostPtr,
                                                         size_t size) {
  DEBUG_PRINT(llvm::formatv("retrieve data of size: {0}", size));
  JitManager::getInstance().moveDataToHostBuffer(hostPtr, size);
}

[[deprecated("Should not use this one, plugin should know less about PJRT")]]
__attribute__((visibility("default")))
PJRT_Buffer *GetPjrtBuffer(void *cpu_ptr) {
  auto &InternalBufferMap = getInternalBufferMap();
  auto it = InternalBufferMap.find(cpu_ptr);
  if (it != InternalBufferMap.end()) {
    std::vector<PJRT_Buffer *> buffers = it->second;
    // FIXME: for test only
    DEBUG_PRINT(llvm::formatv("Buffers size: {}", buffers.size()));
    return buffers[1];
  }
  DEBUG_PRINT("PJRT Buffer Not found!");
  return nullptr;
}

__attribute__((visibility("default"))) void DestroyPjrtBuffer(void *cpu_ptr) {
  JitManager::getInstance().destroyHostBoundBuffers(cpu_ptr);
}

/**
 * JitCode: A function contains the omp::TargetOp with a omp::workdistributeOp
 * inside.
 *
 */
int64_t __botw_jit_code(void *JitCode, int64_t NumArgs, void **TgtArgs,
                        ptrdiff_t *TgtOffsets, void *DeviceArgs,
                        int64_t NumHostArgs, void **ArgBasePtrs, void **ArgPtrs,
                        int64_t *ArgSizes, int64_t *ArgTypes, void **ArgNames) {
  PROFILE_SCOPE("total", Phase::TOTAL);

#ifdef ENABLE_XLA_DEBUG
  // Uncomment if necessary
  // checkDeletgatedLaunchInputs(JitCode, NumArgs, TgtArgs, TgtOffsets,
  //                             NumHostArgs, ArgBasePtrs, ArgPtrs, ArgSizes,
  //                             ArgTypes, ArgNames);
#endif

  // There is an extra pointer added in 2026 Apr. version of LLVM project and it
  // is irrelevant to our project.
  NumArgs -= 1;
  NumHostArgs -= 1;
  ArgSizes -= 1;

  // -----------------------------------------------
  auto &jm = JitManager::getInstance();
  auto l2Cache = jm.getOrCompileKernel(JitCode, NumArgs, TgtArgs, ArgSizes,
                                       ArgPtrs, ArgTypes);
  assert(l2Cache);
  auto newArgs = assembleXLAFuncArgs(l2Cache->kernelFuncTypes, NumHostArgs,
                                     ArgTypes, TgtArgs);
  auto launchArgs = (struct KernelArgs){.inputArgCount = unsigned(NumArgs),
                                        .inputArgs = newArgs.data(),
                                        .outputArgCount = unsigned(NumArgs),
                                        .outputArgs = newArgs.data()};
  JitManager::getInstance().launch(l2Cache->exe, &launchArgs,
                                   l2Cache->kernelFuncStr);
  return 0;
}
}
