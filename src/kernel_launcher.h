#include <cstdint>
#include <string>
#include <sys/types.h>
#include "../third_party/headers/pjrt_c_api.h"
#include "utilities.h"
#include "kernel_pointer_interface.h"
#include "jit-manager.h"

static PJRT_Client* client;
static PJRT_Device* device;

void launchKernel(
  KernelArgs* kernelArgs, 
  const uintptr_t JitCodePtr, 
  const std::string& kernelFuncStr
);

