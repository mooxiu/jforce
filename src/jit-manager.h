#include "../third_party/headers/pjrt_c_api.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <shared_mutex>
#include "kernel_pointer_interface.h"
#include "xla/pjrt/proto/compile_options.pb.h"

class JitManager {
private:
  const PJRT_Api *pjrtApi;
  std::shared_mutex mutex;
  mlir::DenseMap<uintptr_t, PJRT_LoadedExecutable*> XLAKernelsMap;

public:
  JitManager();
  const PJRT_Api* getPJRTApi();
  PJRT_LoadedExecutable* tryGetExecutable(uintptr_t ptr);
  PJRT_LoadedExecutable* compileAndGetExecutable(
    const PJRT_Api *api, 
    PJRT_Client *client,
    const std::string &func_code, 
    KernelArgs* offloadingArgs,
    uintptr_t JitCodePtr);
};

static JitManager jitManager;

__attribute__ ((constructor))
static void initJitMgr() {
  JitManager jitManager;
}


