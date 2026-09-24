#include "jit-manager.h"
#include "../support/profiler.h"
#include "../support/utilities.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

std::unordered_map<void *, std::vector<PJRT_Buffer *>> &getInternalBufferMap() {
  // void* -> pointer in the target. In multi-devices case, the pointer is to a
  // virtual device.
  // FIXME: we should also store the size of the buffer!!!!!
  static std::unordered_map<void *, std::vector<PJRT_Buffer *>> map;
  return map;
}

std::string getPluginPath() {
  const char *path = std::getenv("PJRT_PLUGIN_PATH");

  if (!path || *path == '\0') {
    llvm::report_fatal_error("PJRT_PLUGIN_PATH is not set or is empty.");
  }

  return path;
}

static PJRT_Client *getPJRTClient(const PJRT_Api *api) {
  PJRT_Client_Create_Args args = {};
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  auto error = api->PJRT_Client_Create(&args);
  if (error) {
    std::cerr << "Fail to create client: " << JitManager::getErrMsg(api, error)
              << "\n";
    std::exit(EXIT_FAILURE);
  }
  return args.client;
}

// For filtering out the target device.
static std::string getDeviceDescription(const PJRT_Api *api,
                                        PJRT_Device *device) {
  PJRT_Device_GetDescription_Args args = {
      .struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE,
      .device = device,
  };
  auto err1 = api->PJRT_Device_GetDescription(&args);
  if (err1) {
    std::cerr << "[Error] Fail to get description of the device: "
              << JitManager::getErrMsg(api, err1) << "\n";
    return nullptr;
  }
  PJRT_DeviceDescription_ToString_Args ts_args = {
      .struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE,
      .device_description = args.device_description,
  };
  auto err2 = api->PJRT_DeviceDescription_ToString(&ts_args);
  if (err2) {
    std::cerr << "[Error] Fail to get device description to string: "
              << JitManager::getErrMsg(api, err2) << "\n";
    return nullptr;
  }
  return ts_args.to_string;
}

static TargetDeviceType getTargetDeviceFromEnv() {
  const char *value = std::getenv("JFORCE_TARGET_DEVICE");
  if (!value || *value == '\0') {
    llvm::report_fatal_error("JFORCE_TARGET_DEVICE is not set; "
                             "expected CPU, CUDA, ROCM, or TPU.");
  }
  std::string name(value);
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  if (name == "CPU")
    return TargetDeviceType::CPU;
  if (name == "CUDA")
    return TargetDeviceType::CUDA;
  if (name == "ROCM")
    return TargetDeviceType::ROCM;
  if (name == "TPU")
    return TargetDeviceType::TPU;
  llvm::errs() << "Unsupported device!\n";
  exit(EXIT_FAILURE);
  return TargetDeviceType::INVALID;
}

// Get the target device handle
static llvm::SmallVector<PJRT_Device *>
findDevices(const PJRT_Api *api, PJRT_Client *client,
            const std::string &deviceDescKeyword) {
  PJRT_Client_AddressableDevices_Args device_args = {
      .struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
      .client = client,
  };
  auto err = api->PJRT_Client_AddressableDevices(&device_args);
  if (err) {
    std::cerr << "[Error] Cannot get addressable device: "
              << JitManager::getInstance().getErrMsg(api, err) << "\n";
    std::exit(EXIT_FAILURE);
  }
  if (device_args.num_addressable_devices < 1) {
    std::cerr << "[Error] Cannot find any device!\n";
    std::exit(EXIT_FAILURE);
  }

  llvm::SmallVector<PJRT_Device *> matched_devices;
  for (int i = 0; i < device_args.num_addressable_devices; i++) {
    auto device = device_args.addressable_devices[i];
    std::string tmp = getDeviceDescription(api, device);
    DEBUG_PRINT("Read device description: " + tmp +
                ", and trying to find device: " + deviceDescKeyword);
    std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    DEBUG_PRINT("After lower: " + tmp);
    if (tmp.find(deviceDescKeyword) != std::string::npos) {
      matched_devices.push_back(device);
    }
  }
  DEBUG_PRINT(
      llvm::formatv("Found {0} matched PJRT devices.", matched_devices.size()));
  return matched_devices;
}

JitManager::JitManager() {
  // Initialize context
  mlir::DialectRegistry registry;
  registry
      .insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
              mlir::math::MathDialect, fir::FIROpsDialect, hlfir::hlfirDialect,
              mlir::omp::OpenMPDialect, mlir::scf::SCFDialect,
              mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
              mlir::bufferization::BufferizationDialect,
              mlir::stablehlo::StablehloDialect, mlir::sdy::SdyDialect>();
  mlir::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  this->context.appendDialectRegistry(registry);

  // Iniialize PJRT_API
  SET_XLA_FLAG();
  auto handle_ =
      dlopen(getPluginPath().c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
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
  PJRT_Api *api = get_api_fn();
  PJRT_Plugin_Initialize_Args initArgs = {
      .struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE};
  auto initErr = api->PJRT_Plugin_Initialize(&initArgs);
  assert(!initErr && "Error when plugin initializing!");
  // Theoretically need to close handle_ when exiting, but it will automatically
  // be destroyed when exiting the program so intentionally leave it.
  this->pjrtApi = api;
  this->pjrtClient = getPJRTClient(api);
  this->targetDeviceTy = getTargetDeviceFromEnv();
  // Find all the available devices in the environment.
  switch (this->targetDeviceTy) {
  case TargetDeviceType::CPU:
    DEBUG_PRINT("Trying to get device type: CPU.");
    this->pjrtDevices = findDevices(this->pjrtApi, this->pjrtClient, "cpu");
    break;
  case TargetDeviceType::CUDA:
    DEBUG_PRINT("Trying to get device type: CUDA.");
    this->pjrtDevices = findDevices(this->pjrtApi, this->pjrtClient, "cuda");
    break;
  case TargetDeviceType::ROCM:
    DEBUG_PRINT("Trying to get device type: ROCM.");
    this->pjrtDevices = findDevices(this->pjrtApi, this->pjrtClient, "rocm");
    break;
  case TargetDeviceType::TPU:
    DEBUG_PRINT("Trying to get device type: TPU.");
    this->pjrtDevices = findDevices(this->pjrtApi, this->pjrtClient, "tpu");
    break;
  default:
    DEBUG_PRINT("Trying to get unknown type device, device type is: " +
                std::to_string((int32_t)this->targetDeviceTy));
    std::cerr << "Fail to find device!\n";
    std::exit(EXIT_FAILURE);
  }
  return;
}

JitManager &JitManager::getInstance() {
  static JitManager jitManager;
  return jitManager;
}

const PJRT_Api *JitManager::getPJRTApi() { return this->pjrtApi; }

mlir::MLIRContext *JitManager::getContext() { return &this->context; }

mlir::ModuleOp JitManager::getModuleOp(void *JitCode) {
  auto JitCodePtr = reinterpret_cast<uintptr_t>(JitCode);
  char *JitCodeC = reinterpret_cast<char *>(JitCode);
  // If can found in map, just return a cloned moduleOP
  std::shared_lock<std::shared_mutex> rLock(moduleOpRWMtx);
  auto it = this->moduleOpMap.find(JitCodePtr);
  if (it != moduleOpMap.end()) {
    return it->getSecond()->clone();
  }
  rLock.unlock();

  // Else, need to parse
  mlir::ParserConfig parserConfig(&this->context);
  mlir::OwningOpRef<mlir::ModuleOp> m =
      mlir::parseSourceString<mlir::ModuleOp>(JitCodeC, parserConfig);
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

// The cache key to the compiled kernel function.
// To uniquely identify a kernel function cache, we need to compare all the
// arguments which represents the shape.
//
// Key= JitCodePtr + [ArgSizes[i] + TgtArgs[i]] for i in NumAgrs
//
// JitCodePtr is the pointer to this JIT string captured.
//
// ArgSizes[i]:
// TgtArgs[i]:
llvm::SmallVector<uint64_t, 128> JitManager::getL2JitMetasKey(
    int64_t NumArgs, int64_t *ArgTypes, void **TgtArgs, int64_t *ArgSizes,
    void *JitCode, const llvm::DenseMap<uint32_t, bool> &shapeArgInfoMap) {
  uintptr_t JitCodePtr = reinterpret_cast<std::uintptr_t>(JitCode);
  llvm::SmallVector<uint64_t, 128> key;

  key.push_back(JitCodePtr);
  for (int i = 0; i < NumArgs; i++) {
    key.push_back(ArgSizes[i]);
    assert(shapeArgInfoMap.contains(i));
    if (shapeArgInfoMap.at(i)) {
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
  assert(!destroyErr);
  return;
}

PJRT_LoadedExecutable *
JitManager::compilePJRTExecutable(const std::string &func_code) {
  PJRT_Program program = (struct PJRT_Program){
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .code = (char *)func_code.c_str(),
      .code_size = (size_t)func_code.size(),
      .format = "mlir",
      .format_size = (size_t)4 // Size of 'mlir' 4 chars
  };

  auto getCompileOptionsProto = [&]() -> std::string {
    xla::CompileOptionsProto opts = {};
    opts.set_parameter_is_tupled_arguments(false);
    opts.set_compile_portable_executable(false);
    opts.set_profile_version(1);

    xla::ExecutableBuildOptionsProto *build_opts =
        opts.mutable_executable_build_options();
    build_opts->set_num_replicas(REPLICA_COUNT);
    // build_opts->set_num_partitions(1);

    build_opts->set_num_partitions(PARTITION_COUNT);
    build_opts->set_use_spmd_partitioning(true);
    build_opts->set_use_shardy_partitioner(true);

    build_opts->set_device_memory_size(40LL << 30); // 40 GB

    // ref: https://openxla.org/xla/hlo_dumps
    auto *debug = build_opts->mutable_debug_options();
    debug->set_xla_enable_dumping(true);
    debug->set_xla_dump_hlo_as_text(true);
    debug->set_xla_dump_to(
        "./xla/jforce-sharding" +
        std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count()));
    debug->set_xla_dump_hlo_pass_re("spmd|propagation|sharding-remov|shardy");

    // Special option for CUDA
    if (this->targetDeviceTy == TargetDeviceType::CUDA) {
      // TODO: this might make compiled code slower!!!
      auto debugOptions = build_opts->mutable_debug_options();
      debugOptions->set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(
          true);
      if (const char *cuda_path_env = std::getenv("MY_CUDA_PATH")) {
        debugOptions->set_xla_gpu_cuda_data_dir(cuda_path_env);
      } else {
        // DO NOTHING, this might cause warning
      }
    }

    DEBUG_PRINT("Before compile check: " << opts.DebugString());

    std::string buf;
    // SerializeToString(): This is protobuf's method inherited by
    // `CompileOptionProto`.
    if (!opts.SerializeToString(&buf)) {
      llvm::errs() << "Fail to serialize CompileOptionsProto\n";
      return "";
    }
    return buf;
  };

  // PJRT_Client_Compile will return a LoadedExecutable which can directly be
  // executed.
  auto buf = getCompileOptionsProto();
  PJRT_Client_Compile_Args compile_args = (struct PJRT_Client_Compile_Args){
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .client = this->pjrtClient,
      .program = &program,
      .compile_options = (char *)buf.c_str(),
      .compile_options_size = (size_t)buf.size()};

  auto error = this->pjrtApi->PJRT_Client_Compile(&compile_args);
  if (error) {
    llvm::errs() << "Fail to compile XLA Executable: "
                 << getErrMsg(this->pjrtApi, error) << "\n";
    std::exit(EXIT_FAILURE);
  }

  // Debugging the compiled executable
  // FIXME: hide in if else condition
  PJRT_LoadedExecutable_GetExecutable_Args legeArgs = {
      .struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE,
      .loaded_executable = compile_args.executable};
  auto legeErr = this->pjrtApi->PJRT_LoadedExecutable_GetExecutable(&legeArgs);
  assert(!legeErr && legeArgs.executable);

  PJRT_Executable_NumPartitions_Args npArgs = {
      .struct_size = PJRT_Executable_NumPartitions_Args_STRUCT_SIZE,
      .executable = legeArgs.executable};
  auto npErr = this->pjrtApi->PJRT_Executable_NumPartitions(&npArgs);
  assert(!npErr);
  DEBUG_PRINT(llvm::formatv("After compilation check: Num Partitions: {0}",
                            npArgs.num_partitions));

  PJRT_Executable_NumReplicas_Args nrArgs = {
      .struct_size = PJRT_Executable_NumReplicas_Args_STRUCT_SIZE,
      .executable = legeArgs.executable};
  auto nrErr = this->pjrtApi->PJRT_Executable_NumReplicas(&nrArgs);
  assert(!nrErr);
  DEBUG_PRINT(llvm::formatv("After compilation check: Num Replicas: {0}",
                            nrArgs.num_replicas));

  return compile_args.executable;
}

L1JitMetas *JitManager::tryGetL1JitMetas(void *JitCode) {
  auto JitCodePtr = reinterpret_cast<uintptr_t>(JitCode);
  std::shared_lock<std::shared_mutex> rLock(this->l1JitMetaRWMtx);
  auto it = this->l1JitMetasMap.find(JitCodePtr);
  if (it != this->l1JitMetasMap.end()) {
    return &(it->getSecond());
  }
  return nullptr;
}

void JitManager::saveL1JitMetas(
    void *JitCode, llvm::DenseMap<uint32_t, bool> shapeArgInfoMap) {
  uintptr_t JitCodePtr = reinterpret_cast<uintptr_t>(JitCode);
  std::unique_lock<std::shared_mutex> wLock(this->l1JitMetaRWMtx);
  if (!this->l1JitMetasMap.contains(JitCodePtr)) {
    this->l1JitMetasMap.try_emplace(
        JitCodePtr, L1JitMetas{.shapeArgInfoMap = std::move(shapeArgInfoMap)});
  }
  return;
}

L2JitMetas *
JitManager::tryGetL2JitMetas(llvm::SmallVector<uint64_t, 128> &key) {
  std::shared_lock<std::shared_mutex> rLock(this->l2JitMetaRWMtx);
  auto it = this->l2JitMetasMap.find(key);
  if (it != l2JitMetasMap.end()) {
    return &(it->getSecond());
  }
  return nullptr;
}

// FIXME:  used for demo only
mlir::func::FuncOp hardcodedShard(mlir::func::FuncOp kernelFunc,
                                  mlir::OpBuilder &opBuilder) {
  // add the gloabal constraint
  auto moduleOp = kernelFunc->getParentOfType<mlir::ModuleOp>();
  assert(moduleOp && "suppose module exists");
  auto *ctx = opBuilder.getContext();
  ctx->getOrLoadDialect<mlir::sdy::SdyDialect>();

  mlir::OpBuilder::InsertionGuard lock(opBuilder);
  opBuilder.setInsertionPointToStart(moduleOp.getBody(0));

  auto meshName = "dummyMeshName";
  mlir::sdy::MeshOp::create(
      opBuilder, moduleOp.getLoc(), meshName,
      mlir::sdy::MeshAttr::get(ctx,
                               {mlir::sdy::MeshAxisAttr::get(ctx, "data", 3)}));

  // add sharding or replica to each one
  // for simplicity, we shard on all tensors, replicate on all scalars
  auto displayAr = [](::llvm::ArrayRef<int64_t> arr) -> std::string {
    std::stringstream ss;
    for (int i = 0; i < arr.size(); i++) {
      if (i != 0) {
        ss << "x";
      }
      ss << arr[i];
    }
    return ss.str();
  };

  auto dataAxisAttr = mlir::sdy::AxisRefAttr::get(
      /*context=*/ctx,
      /*name=*/"data",
      /*sub_axis_info=*/{});

  for (unsigned int i = 0; i < kernelFunc.getNumArguments(); i++) {
    auto arg = kernelFunc.getArgument(i);
    auto argTy = llvm::cast<mlir::RankedTensorType>(arg.getType());
    assert(argTy && "we're suppose to be dealing with StableHLO function");
    DEBUG_PRINT(llvm::formatv("ArgIdx: {0}, the rank: {1}, the shape: {2}", i,
                              argTy.getRank(), displayAr(argTy.getShape())));
    if (argTy.getRank() > 0) {
      // Sharding
      kernelFunc.setArgAttr(
          i,
          mlir::sdy::TensorShardingAttr::name, // "sdy.sharding"
          mlir::sdy::TensorShardingAttr::get(
              /*context=*/ctx,
              /*mesh_name=*/meshName,
              /*dim_shardings=*/
              {mlir::sdy::DimensionShardingAttr::get(
                  /*context=*/ctx,
                  /*axes=*/{dataAxisAttr},
                  /*is_closed=*/true, // INFO: I think this means shardy can not
                                      // add new things
                  /*priority=*/std::nullopt)},
              /*replicated_axes=*/{},
              /*unreduced_axes=*/{}));
    } else {
      // Replicate
      kernelFunc.setArgAttr(
          i,
          mlir::sdy::TensorShardingAttr::name, // "sdy.sharding"
          mlir::sdy::TensorShardingAttr::get(
              /*context=*/ctx,
              /*mesh_name=*/meshName,
              /*dim_shardings=*/{},
              /*replicated_axes=*/{dataAxisAttr},
              /*unreduced_axes=*/{}));
    }
  }
  return kernelFunc;
}

L2JitMetas *JitManager::createL2JitMetas(llvm::SmallVector<uint64_t, 128> &key,
                                         mlir::func::FuncOp kernelFunc) {
  PROFILE_SCOPE("createL2JitMetas", Phase::JITCOMPILE);
  mlir::OpBuilder opBuilder(this->getContext());

  // TODO: delete this after testing!
  kernelFunc = hardcodedShard(kernelFunc, opBuilder);
  DEBUG_PRINT_OP(kernelFunc);

  auto moduleOp = kernelFunc->getParentOfType<mlir::ModuleOp>();
  auto kernelFuncStr = getMLIROperationAsString(moduleOp);
  auto exec = this->compilePJRTExecutable(kernelFuncStr);

  std::unique_lock<std::shared_mutex> wLock(this->l2JitMetaRWMtx);
  auto it = this->l2JitMetasMap.find(key);
  if (it != l2JitMetasMap.end()) {
    this->destroyLoadedExecutable(exec);
    return &(it->getSecond());
  }

  auto funcTypes = kernelFunc.getFunctionType().getInputs();
  std::vector<mlir::Type> argTypesVec(funcTypes.begin(), funcTypes.end());

  auto insertedPair = this->l2JitMetasMap.try_emplace(
      key, (L2JitMetas){
               .exe = exec,
               .kernelFuncTypes = std::move(argTypesVec),
               .kernelFuncStr = std::move(kernelFuncStr),
           });
  return &(insertedPair.first->getSecond());
};

PJRT_Client *JitManager::getPJRTClientPointer() { return this->pjrtClient; }
