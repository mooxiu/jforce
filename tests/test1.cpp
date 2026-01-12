#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>


static constexpr char JitCodeExecutorName[] = "__botw_jit_code";
extern "C" int64_t (*JitCodeExecutor)(void *JitCode, int64_t NumArgs,
                                      void **TgtArgs, ptrdiff_t *TgtOffsets,
                                      void *DeviceArgs, int64_t NumHostArgs,
                                      void **ArgBasePtrs, void **ArgPtrs,
                                      int64_t *ArgSizes, int64_t *ArgTypes,
                                      void **ArgNames) = nullptr;


int main(int argc, char** args) {
  auto fName = args[1];

  std::ifstream file(fName);
  if (!file.is_open()) {
    std::cerr << "File is not open!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();

  
  auto handle_ = dlopen("jit-code-executor", RTLD_LAZY | RTLD_LOCAL);
  if (!handle_) {
    std::cerr << "Can not find the .so and open it!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  void* func_ptr = dlsym(handle_, JitCodeExecutorName);
  JitCodeExecutor = reinterpret_cast<decltype(JitCodeExecutor)>(func_ptr);
  auto res = JitCodeExecutor(
    (void*)buffer.str().c_str(), 
    0,  // NumArgs
    {}, // TgtArgs
    {}, // TgtOffsets
    {}, // DeviceArgs
    0, // NumHostArgs
    {}, // ArgBasePtrs
    {}, //ArgPtrs
    0, //ArgSizes
    {}, //ArgTypes
  {}); // ArgNames);
  
  if (res != 0) {
    std::cerr << "Fail In Execution!" << std::endl;
  }
  
  return 0;
}
