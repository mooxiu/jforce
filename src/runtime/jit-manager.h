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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

struct MeshAxis {
  std::string name;
  int ordinal;
  int64_t size;

  MeshAxis(std::string name_, int ordinal_, int64_t size_)
      : name(name_), ordinal(ordinal_), size(size_) {};
};

// Each arg is a nested dimenisonal arrary.
// ArgSharding[i] means how the i-th dimension of the arg sharded or replicaed.
// Because an argument has a
using DimSharding = std::vector<std::reference_wrapper<const MeshAxis>>;
using ArgSharding = std::vector<DimSharding>;
// FIXME: do we have to assign the output sharding?
using ShardingDecision = std::vector<ArgSharding>;

// Profile to be serialized or deserialized, inserted.
// TODO: currently, the key of decisions is written as a void*, this is wrong,
// the pointer can be different in different invokes. I am considering something
// like a hash(ModuleOp)...
struct Profile {
  llvm::DenseMap<void *, ShardingDecision> decisions;
};

struct L1JitMetas {
  llvm::DenseMap<uint32_t, bool> shapeArgInfoMap;
};

struct L2JitMetas {
  PJRT_LoadedExecutable *exe;
  std::vector<mlir::Type> kernelFuncTypes;
  std::string kernelFuncStr;
  ShardingDecision sd;

  L2JitMetas() = delete;
  L2JitMetas(PJRT_LoadedExecutable *exe,
             std::vector<mlir::Type> kernelFuncTypes, std::string kernelFuncStr,
             ShardingDecision sd)
      : exe(exe), kernelFuncTypes(std::move(kernelFuncTypes)),
        kernelFuncStr(std::move(kernelFuncStr)), sd(std::move(sd)) {};
};

class CacheManager {
private:
  mlir::MLIRContext &context_;
  std::shared_mutex l1JitMetaRWMtx;
  llvm::DenseMap<uintptr_t, L1JitMetas> l1JitMetasMap;
  std::shared_mutex l2JitMetaRWMtx;
  llvm::DenseMap<llvm::SmallVector<uint64_t, 128>, L2JitMetas> l2JitMetasMap;
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
  L2JitMetas *insertL2CacheAndReturn(
      const llvm::SmallVector<uint64_t, 128> &key, L2JitMetas &&l2Cache,
      std::function<void(PJRT_LoadedExecutable *)> destroyExecCallback);
};

struct DeviceBufferSlot {
  PJRT_Buffer *bufferPtr;
};

class DeviceBuffersCache {
private:
  std::shared_mutex mtx;
  std::unordered_map<void *, std::vector<DeviceBufferSlot>> deviceBuffersMap;

public:
  std::optional<std::vector<DeviceBufferSlot>> getBuffers(void *dataSrc);
  std::optional<DeviceBufferSlot> getBuffer(void *dataSrc, int devIdx);
  DeviceBufferSlot tryInsertOrUpdate(const PJRT_Api *api, void *dataSrc,
                                     int devIdx, int devCount,
                                     PJRT_Buffer *buffer);
  void deleteEntry(void *dataSrc, const PJRT_Api *api_);
};

class DeviceManager {
  friend class JitManager;

private:
  const PJRT_Api *api_;
  PJRT_Client *client_;
  TargetDeviceType targetDeviceTy_;
  llvm::SmallVector<PJRT_Device *> pjrtDevices_ = {};
  DeviceBuffersCache deviceBuffersCache;

  // forged tgtPtr in host side -> vec{dev0Buffer, dev1Buffer, ....}
  // std::unordered_map<void *, std::vector<PJRT_Buffer *>> deviceBuffersMap;
  PJRT_Buffer *createBufferFromForgedTgtPointers(
      int deviceIdx, void *dataSrc, DType elementDType,
      llvm::ArrayRef<int64_t> subtensorDims,
      llvm::ArrayRef<int64_t> byteStrides, uint32_t offsetInByte);

public:
  DeviceManager(const PJRT_Api *api, PJRT_Client *client,
                TargetDeviceType targetDeviceTy);
  bool invalidateDeviceBuffers(void *forgedTgtPtr, size_t size);
  void destroyHostBoundBuffers(void *hostPtr);
  bool destroyPJRTBuffer(const PJRT_Api *api, PJRT_Buffer *dataPtr);
  PJRT_Buffer *moveDataSegsToDevice(
      const int devIdx, void *dataSrc,
      const llvm::ArrayRef<int64_t> &tensorShape,
      const llvm::ArrayRef<std::pair<size_t, size_t>> &tensorSlices,
      DType elementDType);
  void moveDataToHostBuffer(void *hostPtr, size_t size);
  void moveOutBuffersToDeviceBufferMap(int devIdx, void *argPtr, int argIdx,
                                       PJRT_Buffer **const *outsBuffersList);
  PJRT_Buffer *createLiteralBuffer(int devIdx, const TensorDesc &inputArg);
};

class Sharder {
  friend class JitManager;

public:
  Sharder(uint32_t deviceSize);
  ~Sharder();
  Sharder(const Sharder &) = delete;
  Sharder &operator=(const Sharder &) = delete;

  // heuristicShard accept a moduleOp, and return a sharding decision.
  // The sharding decision is ideally made by using profilings.
  ShardingDecision heuristicShard(mlir::ModuleOp moduleOp);

private:
  uint32_t deviceCount;
  std::string meshName;
  llvm::SmallVector<uint32_t> deviceMesh;
  llvm::SmallVector<MeshAxis> meshAxes;

  Profile *profile = nullptr;
  bool profileChangeFlag = false;
  std::string profilePath = "";

  void addToProfile();
  void serializeProfile();
  Profile *deserializeProfile();
};

class JitManager {
private:
  mlir::MLIRContext context;
  const PJRT_Api *pjrtApi;
  PJRT_Client *pjrtClient;
  TargetDeviceType targetDeviceTy;
  JitManager();
  CacheManager cacheManager;
  DeviceManager deviceManager;
  Sharder sharder;
  L2JitMetas *createL2JitMetas(const llvm::SmallVector<uint64_t, 128> &key,
                               mlir::func::FuncOp kernelFunc);
  mlir::OwningOpRef<mlir::ModuleOp>
  preprocessModuleOp(void *JitCode, int64_t NumArgs, void **TgtArgs,
                     void **ArgPtrs, int64_t *ArgSizes, int64_t *ArgTypes);
  void destroyLoadedExecutable(PJRT_LoadedExecutable *exe);
  mlir::ModuleOp annoateShardyInfo(mlir::ModuleOp moduleOp,
                                   const ShardingDecision &sd);
  PJRT_LoadedExecutable *compilePJRTExecutable(const std::string &func_code);
  void prepareInputBuffers(const TensorDesc *args, const unsigned int count,
                           std::vector<std::vector<PJRT_Buffer *>> &inBuffers,
                           const ShardingDecision &sd);
  void executeOnMultiDevices(PJRT_LoadedExecutable *exe,
                             PJRT_Buffer ***argLists,
                             PJRT_Buffer **const *outLists,
                             const int in_args_count);
  void manageOutBuffers(int inArgsCount, PJRT_Buffer **const *outsBuffersList,
                        TensorDesc *inputArgs, TensorDesc *outputArgs);

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
  bool invalidateCachedDeviceBuffers(void *forgedTgtPtr, size_t size);
  void moveDataToHostBuffer(void *hostPtr, size_t size);

  void destroyHostBoundBuffers(void *hostPtr);

  [[deprecated("should use JitManager::launch")]] void
  launchKernel(PJRT_LoadedExecutable *exec, KernelArgs *kernelArgs,
               const std::string &kernelFuncStr);

  void launch(PJRT_LoadedExecutable *exec, KernelArgs *kernelArgs,
              const std::string &kernelFuncStr, const ShardingDecision &sd);
};
#endif
