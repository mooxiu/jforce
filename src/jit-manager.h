#ifndef JITMANAGER_H
#define JITMANAGER_H
#include "../third_party/headers/pjrt_c_api.h"
#include "utilities.h"
#include "kernel_pointer_interface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <shared_mutex>
#include <vector>
#include "mlir/IR/OwningOpRef.h"


struct L1JitMetas {
  llvm::DenseSet<int> argsIndices;
};

struct L2JitMetas {
  PJRT_LoadedExecutable* exe;
  // mlir::func::FuncOp kernelFunc; 
  std::vector<mlir::Type> kernelFuncTypes;
  std::string kernelFuncStr;
  llvm::DenseMap<unsigned, unsigned> argsIndicesMapping;
};


class JitManager {
private:
  mlir::MLIRContext context;
  const PJRT_Api *pjrtApi;
  PJRT_Client* pjrtClient;
  PJRT_Device* pjrtDevice;


  std::shared_mutex l1JitMetaRWMtx;
  llvm::DenseMap<uintptr_t, L1JitMetas> l1JitMetasMap;

  std::shared_mutex l2JitMetaRWMtx;
  llvm::DenseMap<llvm::SmallVector<uint64_t, 128>, L2JitMetas> l2JitMetasMap; 

  // TODO: add to JitMetasLayer1
  std::shared_mutex moduleOpRWMtx;
  llvm::DenseMap<uintptr_t, mlir::OwningOpRef<mlir::ModuleOp>> moduleOpMap;

  // literal pointer -> buffer
  std::shared_mutex literalPtrBufferCacheRWMtx;
  llvm::DenseMap<std::pair<uintptr_t, DType>, PJRT_Buffer*> literalPtrBufferCache;
  
  PJRT_Device* getPJRTDevice(TargetDevice td);

  PJRT_LoadedExecutable* compilePJRTExecutable(const std::string &func_code, TargetDevice td);
  void destroyLoadedExecutable(PJRT_LoadedExecutable *exe);

 
public:
  JitManager();

  // Making JitManager a singleton
  JitManager(const JitManager&) = delete;
  JitManager& operator=(const JitManager&) = delete;

  static JitManager& getInstance();

  mlir::MLIRContext* getContext();
  const PJRT_Api* getPJRTApi();

  mlir::ModuleOp getModuleOp(uintptr_t JitCodePtr, const char* JitCodeC);

  L1JitMetas* tryGetL1JitMetas(uintptr_t JitCodePtr);
  void saveL1JitMetas(uintptr_t JitCodePtr, llvm::DenseSet<int> argsIndices);
 
  llvm::SmallVector<uint64_t, 128> getL2JitMetasKey(
    int64_t NumArgs, 
    int64_t* ArgTypes, 
    void** TgtArgs, 
    int64_t* ArgSizes, 
    uintptr_t JitCodePtr, 
    llvm::DenseSet<int> argsIndices
  );
  L2JitMetas* tryGetL2JitMetas(llvm::SmallVector<uint64_t, 128>& key); 
  L2JitMetas* createL2JitMetas(
    llvm::SmallVector<uint64_t, 128>& key, 
    mlir::func::FuncOp kernelFunc, 
    llvm::DenseMap<unsigned, unsigned> argsIndicesMapping,
    TargetDevice td
  ); 


  PJRT_Buffer* getLiteralBuffer(PJRT_Device* device, uintptr_t rawPtr, DType dataType);
  bool deleteLiteralBuffer(PJRT_Device* device, uintptr_t rawPtr, DType dataType);

  void launchKernel(PJRT_LoadedExecutable* exec, KernelArgs* kernelArgs, const uintptr_t JitCodePtr, const std::string& kernelFuncStr);

  static std::string getErrMsg(const PJRT_Api *api, PJRT_Error *err);
  static bool checkPJRTError(const PJRT_Api *api, PJRT_Error *err, const std::string &eventName);
};

// // Should be initialized at the beginning
// static JitManager& __dummy = JitManager::getInstance();
//
#endif
