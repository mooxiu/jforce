#include "../third_party/headers/pjrt_c_api.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <shared_mutex>
#include "kernel_pointer_interface.h"
#include "xla/pjrt/proto/compile_options.pb.h"

class JitManager {
private:

  mlir::MLIRContext* context;
  const PJRT_Api *pjrtApi;
  std::shared_mutex mutex;
  mlir::DenseMap<uintptr_t, PJRT_LoadedExecutable*> XLAKernelsMap;

public:
  JitManager();

  // Making JitManager a singleton
  JitManager(const JitManager&) = delete;
  JitManager& operator=(const JitManager&) = delete;


  static JitManager& getInstance();

  mlir::MLIRContext* getContext();

  const PJRT_Api* getPJRTApi();

  PJRT_LoadedExecutable* tryGetExecutable(uintptr_t ptr);

  PJRT_LoadedExecutable* compileAndGetExecutable(
    const PJRT_Api *api, 
    PJRT_Client *client,
    const std::string &func_code, 
    KernelArgs* offloadingArgs,
    uintptr_t JitCodePtr);
};

// Should be initialized at the beginning
static JitManager& __dummy = JitManager::getInstance();

