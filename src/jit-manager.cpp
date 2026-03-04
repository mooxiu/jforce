#include "jit-manager.h"
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <mlir/IR/MLIRContext.h>
#include <string>
#include <utility>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
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
  this->context.loadDialect<
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
  return &this->context;
}

mlir::ModuleOp JitManager::getModuleOp(uintptr_t JitCodePtr, const char* JitCodeC) {
  // If can found in map, just return a cloned moduleOP
  std::shared_lock<std::shared_mutex> rLock(moduleOpRWMtx);
  auto it = this->moduleOpMap.find(JitCodePtr);
  if (it != moduleOpMap.end()) {
    return it->getSecond()->clone();
  }
  rLock.unlock();

  // Else, need to parse
  mlir::ParserConfig parserConfig(&this->context);
  mlir::OwningOpRef<mlir::ModuleOp> m = mlir::parseSourceString<mlir::ModuleOp>(JitCodeC, parserConfig);
  if (!m) {
    std::cerr << "Module not extracted!" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::unique_lock<std::shared_mutex> wLock(moduleOpRWMtx);
  auto it2 = this->moduleOpMap.find(JitCodePtr);
  if (it2 != moduleOpMap.end()) {
    return it2->getSecond()->clone();
  }
  this->moduleOpMap[JitCodePtr] = std::move(m);
  return this->moduleOpMap[JitCodePtr]->clone();
}

// Key= JitCodePtr + [ArgSizes[i] + TgtArgs[i]] for i in NumAgrs 
llvm::SmallVector<uint64_t, 128> getPJRTExecutableKey2(int64_t NumArgs, int64_t* ArgTypes, void** TgtArgs, int64_t* ArgSizes, void* JitCode) {
  llvm::SmallVector<uint64_t, 128> key;

  key.push_back(reinterpret_cast<uintptr_t>(JitCode));

  for (int i = 0; i < NumArgs; i++) {
    key.push_back(ArgSizes[i]);
    if (isLiteralTy(ArgTypes[i])) {
      key.push_back(reinterpret_cast<uintptr_t>(TgtArgs[i]));
    }
  }

  return key;
}

// Executable is uniquely identified by the pointer to the function and the shape of the function.
// Example: 
//  func.func(tensor<1000x1000xf32> arg0, tensor<1000X1000xf32> arg1, tensor<f64> arg2); unitptr_t pointer = 12345678
//  Notice the the rank and element type of each arg will not change
//  We encode it to `12345678,1000,1000,1000,1000,0` (`,` does not exists, just make it easier for eyes to parse) 
//
//  (tensor<f64> is been regarded rank 0, different from tensor<1xf64> which is rank 1)
static llvm::SmallVector<uint8_t> getPJRTExecutableKey(KernelArgs* offloadingArgs, uintptr_t JitCodePtr) {
  // we need 8 uint8_t to represents one uint64_t
  // 256 elements vector can contain 32 numbers without realloc
  // assume all shape size are uint64_t so delimiter is not needed
  llvm::SmallVector<uint8_t, 256> key;

  auto pushToKey = [&](uint64_t value)->void {
    uint8_t* byteArr = reinterpret_cast<uint8_t*>(&value);      
    for (int i = 0; i < sizeof(value); i++) {
      key.push_back(byteArr[i]);
    }
  };
  
  pushToKey(JitCodePtr);
  
  auto argsCount = offloadingArgs->inputArgCount;
  for (int i = 0; i < argsCount; i++) {
    auto argInfo = offloadingArgs->inputArgs[i];
    if (argInfo.rank == 0) {
      pushToKey(0);
      continue;
    } else {
      for (int i = 0; i < argInfo.rank; i++) {
        pushToKey(argInfo.shape[i]);
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
  auto key = getPJRTExecutableKey(offloadingArgs, JitCodePtr); 

  std::shared_lock<std::shared_mutex> rLock(xlaKernelRWMtx);
  auto it = this->XLAKernelsMap.find(key);
  if (it != XLAKernelsMap.end()) {
    return it->second;
  };

  rLock.unlock();
  auto compiled = this->compilePJRTExecutable(api, client, func_code, offloadingArgs, JitCodePtr);
  
  std::unique_lock<std::shared_mutex> wLock(xlaKernelRWMtx);
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
    std::exit(EXIT_FAILURE);
  }
  return compile_args.executable;
}


PJRT_Buffer* JitManager::getLiteralBuffer(
  const PJRT_Api *api, 
  PJRT_Client *client,
  PJRT_Device* device,
  uintptr_t rawPtr, 
  DType dataType
) {
  
  auto key = std::pair(rawPtr, dataType);
  std::shared_lock<std::shared_mutex> rLock(literalPtrBufferCacheRWMtx);
  auto it = literalPtrBufferCache.find(key);
  if (it != literalPtrBufferCache.end()) {
    return it->getSecond();
  }

  // not find
  rLock.unlock();
  int32_t val_i32;
  int64_t val_i64;
  float val_f32;
  double val_f64;
  
  void* host_ptr = nullptr;

  switch (dataType) {
    case DType::I32: {
      val_i32 = static_cast<int32_t>(rawPtr);
      host_ptr = &val_i32;
      break;
    }
    case DType::I64: {
      val_i64 = static_cast<int64_t>(rawPtr);
      host_ptr = &val_i64;
      break;
    }
    case DType::F32: {
      uint32_t low_bits = static_cast<uint32_t>(rawPtr);
      std::memcpy(&val_f32, &low_bits, sizeof(float));
      host_ptr = &val_f32;
      break;
    }
    case DType::F64: {
      std::memcpy(&val_f64, &rawPtr, sizeof(double));
      host_ptr = &val_f64;
      break;
    }
  }

  PJRT_Client_BufferFromHostBuffer_Args buffer_args = {};
  buffer_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  buffer_args.type = [&](){
    if (dataType == DType::F32){
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F32;
    } else if (dataType == DType::F64) {
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F64;
    } else if (dataType == DType::I32) {
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S32;
    } else if (dataType == DType::I64) {
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S64;
    } else {
      std::cerr << "Unknown Buffer Types!\n";
      std::exit(EXIT_FAILURE);
    }
  }();

  int64_t dims[1] = {};
  buffer_args.client = client;
  buffer_args.data = host_ptr;
  buffer_args.dims = dims;
  buffer_args.num_dims = 0; // TODO: should reconsider how to set the size and dimmension for general
  buffer_args.device = device;
 
  auto err = api->PJRT_Client_BufferFromHostBuffer(&buffer_args);
  if (err) {
    std::cerr << "Fail to create literal buffer from host!\n";
    return nullptr;  
  }

  std::unique_lock<std::shared_mutex> wLock(literalPtrBufferCacheRWMtx);
  auto it2 = literalPtrBufferCache.find(key);
  if (it2 != literalPtrBufferCache.end()) {
    PJRT_Buffer_Destroy_Args destroy_args = {
      .struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
      .buffer = buffer_args.buffer
    };
    api->PJRT_Buffer_Destroy(&destroy_args);
    return it2->getSecond();
  }

  literalPtrBufferCache[key] = buffer_args.buffer; 
  return buffer_args.buffer;
}
