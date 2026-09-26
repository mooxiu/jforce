#ifndef JITMANAGER_H
#define JITMANAGER_H

#include "../../third_party/headers/pjrt_c_api.h"
#include "../support/utilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <shared_mutex>
#include <string>

#define PARTITION_COUNT 3
#define REPLICA_COUNT 1

std::unordered_map<void *, std::vector<PJRT_Buffer *>> &getInternalBufferMap();

struct L1JitMetas {
  llvm::DenseMap<uint32_t, bool> shapeArgInfoMap;
};

struct L2JitMetas {
  PJRT_LoadedExecutable *exe;
  std::vector<mlir::Type> kernelFuncTypes;
  std::string kernelFuncStr;
};

class CacheManager {
private:
  mlir::MLIRContext &context_;
  std::shared_mutex l1JitMetaRWMtx;
  llvm::DenseMap<uintptr_t, L1JitMetas> l1JitMetasMap;
  std::shared_mutex l2JitMetaRWMtx;
  llvm::DenseMap<llvm::SmallVector<uint64_t, 128>, L2JitMetas> l2JitMetasMap;
  // NOTE: I feel this can be added to JitMetasLayer1
  std::shared_mutex moduleOpRWMtx;
  llvm::DenseMap<uintptr_t, mlir::OwningOpRef<mlir::ModuleOp>> moduleOpMap;

public:
  CacheManager(mlir::MLIRContext &context) : context_(context) {};
  // Return the parsed moduleOp from JitCode, from second time will return the
  // cached.
  mlir::ModuleOp getModuleOp(void *JitCode);
  L1JitMetas *tryGetL1JitMetas(void *JitCode);
  void saveL1JitMetas(void *JitCode,
                      llvm::DenseMap<uint32_t, bool> shapeArgInfoMap);
  llvm::SmallVector<uint64_t, 128>
  getL2JitMetasKey(int64_t NumArgs, void **TgtArgs, int64_t *ArgSizes,
                   void *JitCode,
                   const llvm::DenseMap<uint32_t, bool> &shapeArgInfoMap);
  L2JitMetas *tryGetL2JitMetas(llvm::SmallVector<uint64_t, 128> &key);
  // L2JitMetas *insertL2CacheAndReturn(llvm::SmallVector<uint64_t, 128> &key,
  //                                    PJRT_LoadedExecutable *exec,
  //                                    std::string_view kernelFuncStr);
  L2JitMetas *insertL2CacheAndReturn(
      llvm::SmallVector<uint64_t, 128> &key, PJRT_LoadedExecutable *exec,
      std::string kernelFuncStr, mlir::func::FuncOp kernelFunc,
      std::function<void (PJRT_LoadedExecutable *)> destroyExec);
};

class DeviceManager {
private:
  const PJRT_Api *api_;
  PJRT_Client *client_;
  TargetDeviceType targetDeviceTy_;
  llvm::SmallVector<PJRT_Device *> pjrtDevices_ = {};
  PJRT_LoadedExecutable *compilePJRTExecutable(const std::string &func_code);

public:
  DeviceManager(const PJRT_Api* api, PJRT_Client* client, TargetDeviceType targetDeviceTy);
  void destroyHostBoundBuffers(void *hostPtr);
  void moveDataToHostBuffer(void *hostPtr, size_t size);
  void launchKernel(PJRT_LoadedExecutable *exec, KernelArgs *kernelArgs,
                    const std::string &kernelFuncStr);
  void launchKernelOnMultiDevices(PJRT_LoadedExecutable *exec,
                                  KernelArgs *kernelArgs,
                                  const std::string &kernelFuncStr);
};

// TODO:
// move cache related fields into a seperate structure
// move device related into sharder (first integrate sharder here)
class JitManager {
private:
  mlir::MLIRContext context;
  const PJRT_Api *pjrtApi;
  PJRT_Client *pjrtClient;
  TargetDeviceType targetDeviceTy;
  JitManager();
  CacheManager cacheManager;
  DeviceManager deviceManager;
  L2JitMetas *createL2JitMetas(llvm::SmallVector<uint64_t, 128> &key,
                               mlir::func::FuncOp kernelFunc);
  mlir::OwningOpRef<mlir::ModuleOp>
  preprocessModuleOp(void *JitCode, int64_t NumArgs, void **TgtArgs,
                     void **ArgPtrs, int64_t *ArgSizes, int64_t *ArgTypes);
  void destroyLoadedExecutable(PJRT_LoadedExecutable *exe);
  PJRT_LoadedExecutable * compilePJRTExecutable(const std::string &func_code);
public:
  // Making JitManager a singleton
  JitManager(const JitManager &) = delete;
  JitManager &operator=(const JitManager &) = delete;
  // Return the singleton instance reference of `JitManager`.
  static JitManager &getInstance();
  static std::string getErrMsg(const PJRT_Api *api, PJRT_Error *err);

  L2JitMetas *getOrCompileKernel(void *JitCode, int64_t NumArgs, void **TgtArgs,
                                 int64_t *ArgSizes, void **ArgPtrs,
                                 int64_t *ArgTypes);

  mlir::func::FuncOp lowerToStableHLO(mlir::ModuleOp moduleOp);
  void moveDataToHostBuffer(void *hostPtr, size_t size);
  void destroyHostBoundBuffers(void *hostPtr);
  void launch(PJRT_LoadedExecutable *exec, KernelArgs *kernelArgs,
              const std::string &kernelFuncStr);
};
#endif
