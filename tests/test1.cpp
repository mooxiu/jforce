#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
  // std::cout << "File Content: " << buffer.str() << std::endl;

  
  auto handle_ = dlopen("libjit-code-executor.so", RTLD_LAZY | RTLD_LOCAL);
  if (!handle_) {
    std::cerr << "Can not find the .so and open it!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  void* func_ptr = dlsym(handle_, JitCodeExecutorName);


  size_t alignment = 64;
  auto alignedSize = [alignment](size_t realSize){
    return realSize%alignment == 0? realSize: realSize + alignment - (realSize%alignment);
  };

  float_t* vec1 = (float_t*)std::aligned_alloc(alignment, alignedSize(sizeof(float_t) * 10));
  float_t* vec2 = (float_t*)std::aligned_alloc(alignment, alignedSize(sizeof(float_t) * 10));
  for (int32_t i = 0; i < 10; i++) {
    vec1[i] = i;
    vec2[i] = 987654;
  }
  void** funcArgs = (void**)std::aligned_alloc(alignment, alignedSize(sizeof(float_t*) * 2));
  funcArgs[0] = (void*)vec1;
  funcArgs[1] = (void*)vec2;
  

  JitCodeExecutor = reinterpret_cast<decltype(JitCodeExecutor)>(func_ptr);
  auto res = JitCodeExecutor(
    (void*)buffer.str().c_str(), 
    2,  // NumArgs
    funcArgs, // TgtArgs
    {}, // TgtOffsets
    {}, // DeviceArgs
    2, // NumHostArgs
    {}, // ArgBasePtrs
    {}, //ArgPtrs
    0, //ArgSizes
    {}, //ArgTypes
  {}); // ArgNames);
  
  if (res != 0) {
    std::cerr << "Fail In Execution!" << std::endl;
  }

  for (int i = 0; i < 10; i++) {
    printf("vec1[i]: %f\n", vec1[i]);
    printf("vec2[i]: %f\n", vec2[i]);
  }

  std::free(vec1);
  std::free(vec2);
  std::free(funcArgs);

  return 0;
}
