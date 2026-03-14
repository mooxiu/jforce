#include "jit-manager.h"
#include "profiler.h"
#include "utilities.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <mlir/IR/MLIRContext.h>
#include <string>
#include <utility>
#include "llvm/ADT/DenseSet.h"
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

static PJRT_Client * getPJRTClient(const PJRT_Api *api) {
  PJRT_Client_Create_Args args = {};
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  auto error = api->PJRT_Client_Create(&args);
  if (error) {
    std::cerr << "Fail to create client!\n";
    std::exit(EXIT_FAILURE);
  }
  return args.client;
}

// For filtering out the target device.
static std::string getDeviceDescription(const PJRT_Api *api, PJRT_Device *device) {
  PJRT_Device_GetDescription_Args args = {
    .struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE,
    .device = device,
  };
  auto err1 = api->PJRT_Device_GetDescription(&args);
  if (err1) {
    logger::Log("Fail to get description of device: " + JitManager::getErrMsg(api, err1),
                logLevel::ERROR);
    return nullptr;
  }
  PJRT_DeviceDescription_ToString_Args ts_args = {
    .struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE,
    .device_description = args.device_description,
  };
  auto err2 = api->PJRT_DeviceDescription_ToString(&ts_args);
  if (err2) {
    logger::Log("Fail to get device description to string: " +
                    JitManager::getErrMsg(api, err2),
                logLevel::ERROR);
    return nullptr;
  }
  return ts_args.to_string;
}

// Get the target device handle
static PJRT_Device *findDevice(const PJRT_Api *api, PJRT_Client *client,
                        const std::string &deviceDescKeyword) {
  PJRT_Client_AddressableDevices_Args device_args = {
    .struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
    .client = client,
  };
  auto err = api->PJRT_Client_AddressableDevices(&device_args);
  if (!JitManager::checkPJRTError(api, err, "Find Device")) {
    return nullptr;
  }
  if (device_args.num_addressable_devices < 1) {
    logger::Log("Cannot find any device!", logLevel::ERROR);
    return nullptr;
  }

  int chosen_device_idx = -1;
  std::string desc = ""; // for logging purpose
  for (int i = 0; i < device_args.num_addressable_devices; i++) {
    std::string tmp = getDeviceDescription(api, device_args.addressable_devices[i]);
    std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                   [](auto c) { return std::tolower(c); });
    if (tmp.find(deviceDescKeyword) != std::string::npos) {
      chosen_device_idx = i;
      desc = tmp;
      break;
    }
  }
  if (chosen_device_idx == -1) {
    logger::Log("Fail to find " + deviceDescKeyword + " device!",
                logLevel::ERROR);
    return nullptr;
  }
  return device_args.addressable_devices[chosen_device_idx];
}

PJRT_Device* JitManager::getPJRTDevice(TargetDevice td) {
  if (this->pjrtDevice) {
    return this->pjrtDevice;
  }
  if (td == TargetDevice::CPU) {
    this->pjrtDevice = findDevice(this->pjrtApi, this->pjrtClient, "cpu");
  } else if (td == TargetDevice::CUDA){
    this->pjrtDevice = findDevice(this->pjrtApi, this->pjrtClient, "cuda");
  } else {
    return nullptr;
  }
  return this->pjrtDevice;
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
  SET_XLA_FLAG();
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
  this->pjrtClient = getPJRTClient(api);
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
llvm::SmallVector<uint64_t, 128> JitManager::getL2JitMetasKey(
  int64_t NumArgs, 
  int64_t* ArgTypes, 
  void** TgtArgs, 
  int64_t* ArgSizes, 
  uintptr_t JitCodePtr, 
  llvm::DenseSet<int> argsIndices
){
  llvm::SmallVector<uint64_t, 128> key;

  key.push_back(JitCodePtr);
  for (int i = 0; i < NumArgs; i++) {
    key.push_back(ArgSizes[i]);
    if (argsIndices.contains(i) && isLiteralTy(ArgTypes[i])) {
      key.push_back(reinterpret_cast<uintptr_t>(TgtArgs[i]));
    }
  }
  return key;
}

void JitManager::destroyLoadedExecutable(PJRT_LoadedExecutable *exe) {
  PJRT_LoadedExecutable_Destroy_Args ledargs = {
    .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
    .executable = exe,
  };
  auto destroyErr = this->pjrtApi->PJRT_LoadedExecutable_Destroy(&ledargs);
  checkPJRTError(this->pjrtApi, destroyErr, "Destroy LoadedExecutable");
  return;
}

PJRT_LoadedExecutable* JitManager::compilePJRTExecutable(const std::string &func_code, TargetDevice td){
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
    build_opts->set_device_memory_size(40LL << 30); // 40 GB

    // Special option for CUDA
    if (td == TargetDevice::CUDA) {
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
    .client = this->pjrtClient,
    .program = &program,
    .compile_options = (char *)buf.c_str(),
    .compile_options_size = (size_t)buf.size()
  };

  auto error = this->pjrtApi->PJRT_Client_Compile(&compile_args);
  if (error) {
    llvm::errs() << "Fail to compile XLA Executable!\n";
    std::exit(EXIT_FAILURE);
  }
  return compile_args.executable;
}

L1JitMetas* JitManager::tryGetL1JitMetas(uintptr_t JitCodePtr){
  std::shared_lock<std::shared_mutex> rLock(this->l1JitMetaRWMtx);
  auto it = this->l1JitMetasMap.find(JitCodePtr);
  if (it != this->l1JitMetasMap.end()) {
    return &(it->getSecond());
  }
  return nullptr;
}

void JitManager::saveL1JitMetas(uintptr_t JitCodePtr, llvm::DenseSet<int> argsIndicesToSave){
  std::unique_lock<std::shared_mutex> wLock(this->l1JitMetaRWMtx);
  if (!this->l1JitMetasMap.contains(JitCodePtr)) {
    this->l1JitMetasMap.try_emplace(JitCodePtr, L1JitMetas{.argsIndices = std::move(argsIndicesToSave)});
  }
  return;
}

L2JitMetas* JitManager::tryGetL2JitMetas(llvm::SmallVector<uint64_t, 128>& key){
  std::shared_lock<std::shared_mutex> rLock(this->l2JitMetaRWMtx);
  auto it = this->l2JitMetasMap.find(key);
  if (it != l2JitMetasMap.end()) {
    return &(it->getSecond());
  }
  return nullptr;
}

L2JitMetas* JitManager::createL2JitMetas(
  llvm::SmallVector<uint64_t, 128>& key, 
  mlir::func::FuncOp kernelFunc, 
  llvm::DenseMap<unsigned, unsigned> argsIndicesMapping,
  TargetDevice td
){
  PROFILE_SCOPE("createL2JitMetas", Phase::JITCOMPILE) 

  auto kernelFuncStr = getMLIROperationAsString(kernelFunc);
  auto exec = this->compilePJRTExecutable(kernelFuncStr, td);

  std::unique_lock<std::shared_mutex> wLock(this->l2JitMetaRWMtx);
  auto it = this->l2JitMetasMap.find(key);
  if (it != l2JitMetasMap.end()) {
    this->destroyLoadedExecutable(exec);
    return &(it->getSecond());
  }

  auto funcTypes = kernelFunc.getFunctionType().getInputs();
  std::vector<mlir::Type> argTypesVec(funcTypes.begin(), funcTypes.end());

  auto insertedPair = this->l2JitMetasMap.try_emplace(
    key,
    (L2JitMetas){
      .exe = exec,
      .kernelFuncTypes = std::move(argTypesVec),
      .kernelFuncStr = std::move(kernelFuncStr),
      .argsIndicesMapping = std::move(argsIndicesMapping)
    }
  );
  return &(insertedPair.first->getSecond());
}; 

