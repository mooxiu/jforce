#include "jit-manager.h"
#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <mlir/IR/MLIRContext.h>
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "stablehlo/dialect/StablehloOps.h"


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

// map elements will not be deleted during the execution, so we can check without lock at first
PJRT_LoadedExecutable* JitManager::tryGetExecutable(uintptr_t ptr){
  if (this->XLAKernelsMap.contains(ptr)) {
    return this->XLAKernelsMap.at(ptr);
  }      
  return nullptr;
}


// Each time when compiling an executable, we should also store it in the map for later usage.
PJRT_LoadedExecutable* JitManager::compileAndGetExecutable(
  const PJRT_Api *api, 
  PJRT_Client *client,
  const std::string &func_code, 
  KernelArgs* offloadingArgs,
  uintptr_t JitCodePtr
){
  std::shared_lock<std::shared_mutex> lock(this->mutex);
  if (this->XLAKernelsMap.contains(JitCodePtr)) {
    return this->XLAKernelsMap.at(JitCodePtr);
  }

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
  XLAKernelsMap.insert(std::pair(JitCodePtr, compile_args.executable));
  return compile_args.executable;
}


