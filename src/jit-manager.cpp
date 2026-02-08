#include "jit-manager.h"
#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <mlir/IR/MLIRContext.h>
#include <string>
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/pjrt/proto/compile_options.pb.h"


std::string getPluginPath() {
// DEFAULT_PJRT_PLUGIN_PATH should be defined in CMake
#ifdef DEFAULT_PJRT_PLUGIN_PATH
  return DEFAULT_PJRT_PLUGIN_PATH;
#else
  throw std::runtime_error(
      "PJRT plugin path not found. Please set PJRT_PLUGIN_PATH.");
#endif
}

JitManager::JitManager() {
  // Initialize context
  this->context->loadDialect<
    mlir::func::FuncDialect,
    mlir::omp::OpenMPDialect,
    fir::FIROpsDialect, 
    hlfir::hlfirDialect,
    mlir::arith::ArithDialect, 
    mlir::stablehlo::StablehloDialect>();


  // Iniialize PJRT_API
  auto handle_ = dlopen(getPluginPath().c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  if (!handle_) {
    std::cerr << "error loading plugin: " << dlerror() << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // follow the example of `man dlopen`
  auto get_api_fn = (PJRT_Api * (*)()) dlsym(handle_, "GetPjrtApi");
  if (!get_api_fn) {
    std::cerr << "error finding GetPjrtApi: " << dlerror() << std::endl;
    std::exit(EXIT_FAILURE);
  }
  PJRT_Api* api = get_api_fn();
  PJRT_Plugin_Initialize_Args initArgs = {};
  initArgs.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  auto initErr = api->PJRT_Plugin_Initialize(&initArgs);
  // Theoretically need to close handle_ when exiting, but it will automatically be destroyed when exiting the program so intentionally leave it.
  this->pjrtApi = api;
}

JitManager& JitManager::getInstance() {
  static JitManager jitManager;
  return jitManager;
}

const PJRT_Api* JitManager::getPJRTApi() {
  return this->pjrtApi;
}

mlir::MLIRContext* JitManager::getContext() {
  return this->context;
}


// Executable is uniquely identified by the pointer to the function and the shape of the function.
// Example: 
//  func.func(tensor<1000x1000xf32> arg0, tensor<1000X1000xf32> arg1, tensor<f64> arg2); unitptr_t pointer = 12345678
//  Notice the the rank and element type of each arg will not change
//  We encode it to `12345678:1000:1000:1000:1000:0` 
//
//  (tensor<f64> is been regarded rank 0, different from tensor<1xf64> which is rank 1)
std::string getPJRTExecutableKey(KernelArgs* offloadingArgs, uintptr_t JitCodePtr) {
  std::string key;
  std::string delimiter = ":"; 
  key.reserve(256); 

  key.append(std::to_string(JitCodePtr));
  
  auto argsCount = offloadingArgs->inputArgCount;
  for (int i = 0; i < argsCount; i++) {
    auto argInfo = offloadingArgs->inputArgs[i];
    key.append(delimiter);   

    if (argInfo.rank == 0) {
      key.append(":0"); 
      continue;
    } else {
      for (int i = 0; i < argInfo.rank; i++) {
        key.append(":");  
        key.append(std::to_string(argInfo.shape[i]));
      }
    }
  }

  return key; 
}


// Simple RWLock implementation.
// One problem is that multiple threads might be compiling the same executable and could waste some CPU cycle.
// TODO: use an extra set to control which executable is being compiled. 
PJRT_LoadedExecutable* JitManager::getPJRTExecutable( 
    const PJRT_Api *api, 
    PJRT_Client *client,
    const std::string &func_code, 
    KernelArgs* offloadingArgs,
    uintptr_t JitCodePtr
){
  auto key = this->getPJRTExecutableKey(offloadingArgs, JitCodePtr); 

  std::shared_lock<std::shared_mutex> rLock(rwmtx);
  auto it = this->XLAKernelsMap.find(key);
  if (it != XLAKernelsMap.end()) {
    return it->second;
  };

  rLock.unlock();
  auto compiled = this->compilePJRTExecutable(api, client, func_code, offloadingArgs, JitCodePtr);
  
  std::unique_lock<std::shared_mutex> wLock(rwmtx);
  // other threads might alredy compiled and insert this one
  auto it2 = this->XLAKernelsMap.find(key);
  if (it2 != XLAKernelsMap.end()) {
    return it2->second;
  };
  // if really not found, insert
  XLAKernelsMap.insert(std::pair(key, compiled)); 
  return compiled;
}

PJRT_LoadedExecutable* JitManager::compilePJRTExecutable(
  const PJRT_Api *api, 
  PJRT_Client *client,
  const std::string &func_code, 
  KernelArgs* offloadingArgs,
  uintptr_t JitCodePtr
){
  PJRT_Program program = (struct PJRT_Program){
    .struct_size = PJRT_Program_STRUCT_SIZE,
    .code = (char*) func_code.c_str(),
    .code_size = (size_t)func_code.size(),
    .format = "mlir",
    .format_size = (size_t) 4 // Size of 'mlir' 4 chars 
  }; 

  auto getCompileOptionsProto = [&]() -> std::string {
    xla::CompileOptionsProto opts = {};
    opts.set_parameter_is_tupled_arguments(false);
    opts.set_compile_portable_executable(false);
    opts.set_profile_version(1);

    xla::ExecutableBuildOptionsProto *build_opts = opts.mutable_executable_build_options();
    build_opts->set_num_replicas(1);
    build_opts->set_num_partitions(1);

    // Special option for CUDA
    if (offloadingArgs->targetDevice == TargetDevice::CUDA) {
      // TODO: this might make compiled code slower!!!
      auto debugOptions = build_opts->mutable_debug_options();
      debugOptions->set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(true);
      if (const char* cuda_path_env = std::getenv("MY_CUDA_PATH")) {
        debugOptions->set_xla_gpu_cuda_data_dir(cuda_path_env);
      } else {
        // DO NOTHING, this might cause warning
      } 
    }    

    std::string buf;
    // SerializeToString(): This is protobuf's method inherited by `CompileOptionProto`.
    if (!opts.SerializeToString(&buf)) {
      llvm::errs() << "Fail to serialize CompileOptionsProto\n";
      return "";
    }
    return buf;
  };

  // It seems PJRT_Client_Compile will also help to load the execute
  auto buf =  getCompileOptionsProto();
  PJRT_Client_Compile_Args compile_args = (struct PJRT_Client_Compile_Args){
    .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
    .client = client,
    .program = &program,
    .compile_options = (char *)buf.c_str(),
    .compile_options_size = (size_t)buf.size()
  };

  auto error = api->PJRT_Client_Compile(&compile_args);
  if (error) {
    llvm::errs() << "Fail to compile XLA Executable!\n";
    return nullptr;
  }
  return compile_args.executable;
}


