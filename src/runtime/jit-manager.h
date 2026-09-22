#ifndef JITMANAGER_H
#define JITMANAGER_H

#include "../../third_party/headers/pjrt_c_api.h"
#include "../support/utilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <shared_mutex>

std::unordered_map<void *, std::vector<PJRT_Buffer *>> &getInternalBufferMap();

struct L1JitMetas {
  llvm::DenseMap<uint32_t, bool> shapeArgInfoMap;
};

struct L2JitMetas {
  PJRT_LoadedExecutable *exe;
  std::vector<mlir::Type> kernelFuncTypes;
  std::string kernelFuncStr;
};

class JitManager {
private:
  mlir::MLIRContext context;
  const PJRT_Api *pjrtApi;
  PJRT_Client *pjrtClient;
  // Should not lazily load, it can cause some latency.
  llvm::SmallVector<PJRT_Device *> pjrtDevices = {};
  TargetDeviceType targetDeviceTy;

  std::shared_mutex l1JitMetaRWMtx;
  llvm::DenseMap<uintptr_t, L1JitMetas> l1JitMetasMap;
  std::shared_mutex l2JitMetaRWMtx;
  llvm::DenseMap<llvm::SmallVector<uint64_t, 128>, L2JitMetas> l2JitMetasMap;
  // NOTE: I feel this can be added to JitMetasLayer1
  std::shared_mutex moduleOpRWMtx;
  llvm::DenseMap<uintptr_t, mlir::OwningOpRef<mlir::ModuleOp>> moduleOpMap;

  PJRT_LoadedExecutable *compilePJRTExecutable(const std::string &func_code);
  void destroyLoadedExecutable(PJRT_LoadedExecutable *exe);

public:
  JitManager();

  // Making JitManager a singleton
  JitManager(const JitManager &) = delete;
  JitManager &operator=(const JitManager &) = delete;

  // Return the singleton instance reference of `JitManager`.
  static JitManager &getInstance();

  // Return some private fields with public methods.
  mlir::MLIRContext *getContext();
  const PJRT_Api *getPJRTApi();
  PJRT_Client *getPJRTClientPointer();

  // Return the parsed moduleOp from JitCode, from second time will return the
  // cached.
  mlir::ModuleOp getModuleOp(void *JitCode);

  L1JitMetas *tryGetL1JitMetas(void *JitCode);
  void saveL1JitMetas(void *JitCode,
                      llvm::DenseMap<uint32_t, bool> shapeArgInfoMap);
  llvm::SmallVector<uint64_t, 128>
  getL2JitMetasKey(int64_t NumArgs, int64_t *ArgTypes, void **TgtArgs,
                   int64_t *ArgSizes, void *JitCode,
                   const llvm::DenseMap<uint32_t, bool> &shapeArgInfoMap);
  L2JitMetas *tryGetL2JitMetas(llvm::SmallVector<uint64_t, 128> &key);
  L2JitMetas *createL2JitMetas(llvm::SmallVector<uint64_t, 128> &key,
                               mlir::func::FuncOp kernelFunc);

  void launchKernel(PJRT_LoadedExecutable *exec, KernelArgs *kernelArgs,
                    const std::string &kernelFuncStr);
  void launchKernelOnMultiDevices(PJRT_LoadedExecutable *exec,
                                  KernelArgs *kernelArgs,
                                  const std::string &kernelFuncStr);

  static std::string getErrMsg(const PJRT_Api *api, PJRT_Error *err);
};
#endif
