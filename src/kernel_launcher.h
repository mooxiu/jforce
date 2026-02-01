#include "kernel_pointer_interface.h"
#include <cstdint>
#include <string>
#include <sys/types.h>

void launchKernel(
  KernelArgs* kernelArgs, 
  const uintptr_t JitCodePtr, 
  const std::string& kernelFuncStr
);

