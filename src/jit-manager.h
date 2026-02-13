#include "../third_party/headers/pjrt_c_api.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <cstdint>
#include <shared_mutex>
#include "kernel_pointer_interface.h"

class JitManager {
private:

  mlir::MLIRContext* context;
 
  const PJRT_Api *pjrtApi;

  std::shared_mutex moduleOpRWMtx;
  llvm::DenseMap<uintptr_t, mlir::ModuleOp> moduleOpMap;

  std::shared_mutex xlaKernelRWMtx;
  llvm::StringMap<PJRT_LoadedExecutable*> XLAKernelsMap;
  
  PJRT_LoadedExecutable* compilePJRTExecutable(
    const PJRT_Api *api, 
    PJRT_Client *client,
    const std::string &func_code, 
    KernelArgs* offloadingArgs,
    uintptr_t JitCodePtr);

  std::string getPJRTExecutableKey(KernelArgs* offloadingArgs, uintptr_t JitCodePtr);

public:
  JitManager();

  // Making JitManager a singleton
  JitManager(const JitManager&) = delete;
  JitManager& operator=(const JitManager&) = delete;


  static JitManager& getInstance();

  mlir::MLIRContext* getContext();

  const PJRT_Api* getPJRTApi();

  mlir::ModuleOp getModuleOp(uintptr_t JitCodePtr, const char* JitCodeC);

  PJRT_LoadedExecutable* getPJRTExecutable(
    const PJRT_Api *api, 
    PJRT_Client *client,
    const std::string &func_code, 
    KernelArgs* offloadingArgs,
    uintptr_t JitCodePtr);
};

// Should be initialized at the beginning
static JitManager& __dummy = JitManager::getInstance();

