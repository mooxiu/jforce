#include "jit-manager.h"
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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/WalkResult.h"
#include "pipelines.h"
#include "runtime/sharding/sharding-codec.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define JIT_LITERAL_VAL_ATTR_NAME "jit.literal_val"
#define JIT_ARGS_MAPPING_ATTR_NAME "jit.args_mapping"
#define JIT_SHAPE_ARG_ATTR_NAME "jit.shape_arg"

#ifdef ENABLE_XLA_DEBUG
#define PRINT_PASS()                                                           \
  llvm::errs() << "Pass pipeline: ";                                           \
  pm.printAsTextualPipeline(llvm::errs());                                     \
  llvm::errs() << "\n";                                                        \
  ctx.disableMultithreading();                                                 \
  pm.enableIRPrinting()
#else
#define PRINT_PASS()
#endif

static std::string getPluginPath() {
  const char *path = std::getenv("PJRT_PLUGIN_PATH");

  if (!path || *path == '\0') {
    llvm::report_fatal_error("PJRT_PLUGIN_PATH is not set or is empty.");
  }

  return path;
}

static const PJRT_Api *loadPjrtAPi() {
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
  return api;
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

static std::string getProfileFileFromEnv() {
  const char *path = std::getenv("JFORCE_PROFILE_FILE");
  if (!path || *path == '\0') {
    llvm::report_fatal_error("JFORCE_PROFILE_FILE is missing, will use default!");
  }
  std::string name(path);
  return name;
}

JitManager::JitManager()
    : pjrtApi(loadPjrtAPi()), pjrtClient(getPJRTClient(pjrtApi)),
      targetDeviceTy(getTargetDeviceFromEnv()), cacheManager(context),
      deviceManager(pjrtApi, pjrtClient, targetDeviceTy),
      sharder(deviceManager.pjrtDevices_.size(), getProfileFileFromEnv()) {
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
  SET_XLA_FLAG();
}

JitManager &JitManager::getInstance() {
  static JitManager jitManager;
  return jitManager;
}

std::string JitManager::getErrMsg(const PJRT_Api *api, PJRT_Error *err) {
  PJRT_Error_GetCode_Args code_args = {};
  code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  code_args.error = err;

  api->PJRT_Error_GetCode(&code_args);

  PJRT_Error_Message_Args msg_args = {};
  msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  msg_args.error = err;
  api->PJRT_Error_Message(&msg_args);
  std::string s(msg_args.message);

  PJRT_Error_Destroy_Args destroy_args = {};
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.error = err;

  api->PJRT_Error_Destroy(&destroy_args);
  return s;
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
    build_opts->set_num_replicas(1);
    // build_opts->set_num_partitions(1);

    build_opts->set_num_partitions(deviceManager.pjrtDevices_.size());
    build_opts->set_use_spmd_partitioning(true);
    build_opts->set_use_shardy_partitioner(true);
    build_opts->allow_spmd_sharding_propagation_to_output();
    build_opts->allow_spmd_sharding_propagation_to_parameters();

    build_opts->set_device_memory_size(40LL << 30); // 40 GB

    // Do not touch debug_options: a proto we create starts with every field
    // 0, and XLA uses it as is instead of its defaults. Set dump and CUDA
    // options through XLA_FLAGS (see scripts/utils.py).

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

  DEBUG_PRINT("This is the function we're going to compile: " + func_code);

  auto error = this->pjrtApi->PJRT_Client_Compile(&compile_args);
  if (error) {
    llvm::errs() << "Fail to compile XLA Executable: "
                 << getErrMsg(this->pjrtApi, error) << "\n";
    std::exit(EXIT_FAILURE);
  }

#ifdef ENABLE_XLA_DEBUG
  // Debugging the compiled executable
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
#endif

  return compile_args.executable;
}

mlir::func::FuncOp JitManager::lowerToStableHLO(mlir::ModuleOp moduleOp) {
  // Lower JItCode to StableHLO
  mlir::PassManager pm(&this->context);
  // INFO: uncomment print pass when we need to debug the IR transformation
  auto &ctx = this->context;
  PRINT_PASS();
  pm.enableCrashReproducerGeneration("./crash_repro.mlir");

  bool hasWorkdistribute = false;
  moduleOp.walk([&](omp::WorkdistributeOp) {
    hasWorkdistribute = true;
    return mlir::WalkResult::interrupt();
  });
  createLowerToStableHLOPassPipeline(pm, hasWorkdistribute);
  if (mlir::failed(pm.run(moduleOp))) {
    llvm::errs() << "MLIR Pass Pipeline failed!\n";
    std::exit(EXIT_FAILURE);
  }
  auto kernelFunc = moduleOp.lookupSymbol<func::FuncOp>("main");
  assert(kernelFunc && "Kernel Func should be renamed as main!\n");
  return kernelFunc;
};

// Return OwningOpRef instead of a raw mlir::ModuleOp to keep the onwership
mlir::OwningOpRef<mlir::ModuleOp>
JitManager::preprocessModuleOp(void *JitCode, int64_t NumArgs, void **TgtArgs,
                               void **ArgPtrs, int64_t *ArgSizes,
                               int64_t *ArgTypes) {
  struct jitArg {
    void *hostPtr;
    void *tgtPtr;
    int64_t size;
    bool isLiteral;
  };

  auto packJitArg = [&]() {
    llvm::SmallVector<jitArg> args;
    args.resize(NumArgs);

    for (int i = 0; i < NumArgs; i++) {
      args[i] = jitArg{
          .hostPtr = ArgPtrs[i],
          .tgtPtr = TgtArgs[i],
          .size = ArgSizes[i],
          .isLiteral = isLiteralTy(ArgTypes[i]),
      };
    };
    return args;
  };

  auto insertJitInfo = [&](mlir::OpBuilder &builder, func::FuncOp kernelFunc,
                           llvm::SmallVector<jitArg> args) {
    auto ctx = builder.getContext();
    auto attrName = builder.getStringAttr(JIT_LITERAL_VAL_ATTR_NAME);
    for (int i = 0; i < args.size(); i++) {
      if (args[i].isLiteral) {
        std::uintptr_t literalAddr =
            reinterpret_cast<std::uintptr_t>(args[i].hostPtr);
        auto attr = IntegerAttr::get(IntegerType::get(ctx, sizeof(void *) * 8),
                                     literalAddr);
        kernelFunc.setArgAttr(i, attrName, attr);
      }
    }
    return;
  };

  // Store pair of <arg index, is shape arg>
  auto getShapeArgInfoMap =
      [&](func::FuncOp funcOp) -> llvm::DenseMap<uint32_t, bool> {
    llvm::DenseMap<uint32_t, bool> shapeArgInfoMap;
    for (uint32_t i = 0; i < funcOp.getNumArguments(); i++) {
      if (funcOp.getArgAttrOfType<UnitAttr>(i, JIT_SHAPE_ARG_ATTR_NAME)) {
        shapeArgInfoMap[i] = true;
      } else {
        shapeArgInfoMap[i] = false;
      }
    }
    return shapeArgInfoMap;
  };

  mlir::OpBuilder builder(&this->context);
  // Use OweningOpRef so RAII can help to destroy the tree
  mlir::OwningOpRef<mlir::ModuleOp> moduleOpRef =
      cacheManager.getModuleOp(JitCode);
  auto kernel = moduleOpRef->lookupSymbol<func::FuncOp>("kernel");
  assert(kernel && "FuncOp with name kernel should exist!");
  auto jitArgs = packJitArg();
  insertJitInfo(builder, kernel, jitArgs);
  return moduleOpRef;
}

mlir::ModuleOp JitManager::annotateDeviceMeshInfo(mlir::ModuleOp moduleOp) {
  OpBuilder opBuilder(&context);
  context.getOrLoadDialect<mlir::sdy::SdyDialect>();

  mlir::OpBuilder::InsertionGuard lock(opBuilder);
  opBuilder.setInsertionPointToStart(moduleOp.getBody(0));

  // Mesh Constraints
  llvm::SmallVector<sdy::MeshAxisAttr> axes;
  axes.resize(sharder.meshAxes.size());
  for (int i = 0; i < sharder.meshAxes.size(); i++) {
    MeshAxis axis = sharder.meshAxes[i];
    axes[i] = sdy::MeshAxisAttr::get(&context, axis.name, axis.size);
  }
  mlir::sdy::MeshOp::create(opBuilder, moduleOp.getLoc(), sharder.meshName,
                            mlir::sdy::MeshAttr::get(&context, axes));
  return moduleOp;
}

L2JitMetas *
JitManager::createL2JitMetas(const llvm::SmallVector<uint64_t, 128> &key,
                             mlir::func::FuncOp kernelFunc,
                             size_t kernelFingerprint) {
  mlir::OpBuilder opBuilder(&this->context);
  auto moduleOp = kernelFunc->getParentOfType<mlir::ModuleOp>();
  moduleOp = annotateDeviceMeshInfo(moduleOp);
  auto plan = buildGraph(kernelFunc).tune(sharder.meshAxes);
  KernelIOShardings ioShardings =
      annotatePlan(kernelFunc, plan, sharder.meshName);
  auto kernelFuncStr = getMLIROperationAsString(moduleOp);
  auto exec = this->compilePJRTExecutable(kernelFuncStr);

  auto funcTypes = kernelFunc.getFunctionType().getInputs();
  std::vector<mlir::Type> argTypesVec(funcTypes.begin(), funcTypes.end());
  L2JitMetas l2Cache(exec, std::move(argTypesVec), std::move(kernelFuncStr),
                     std::move(ioShardings), kernelFingerprint);
  return cacheManager.insertL2CacheAndReturn(
      key, std::move(l2Cache),
      [this](PJRT_LoadedExecutable *e) { this->destroyLoadedExecutable(e); });
};

L2JitMetas *JitManager::getOrCompileKernel(void *JitCode, int64_t NumArgs,
                                           void **TgtArgs, int64_t *ArgSizes,
                                           void **ArgPtrs, int64_t *ArgTypes) {
  auto getShapeArgInfoMap =
      [&](func::FuncOp funcOp) -> llvm::DenseMap<uint32_t, bool> {
    llvm::DenseMap<uint32_t, bool> shapeArgInfoMap;
    for (uint32_t i = 0; i < funcOp.getNumArguments(); i++) {
      if (funcOp.getArgAttrOfType<UnitAttr>(i, JIT_SHAPE_ARG_ATTR_NAME)) {
        shapeArgInfoMap[i] = true;
      } else {
        shapeArgInfoMap[i] = false;
      }
    }
    return shapeArgInfoMap;
  };

  auto *l1Cache = cacheManager.tryGetL1JitMetas(JitCode);
  if (l1Cache) {
    auto l2CacheKey = cacheManager.getL2JitMetasKey(
        NumArgs, TgtArgs, ArgSizes, JitCode, l1Cache->shapeArgInfoMap);
    auto *l2Cache = cacheManager.tryGetL2JitMetas(l2CacheKey);
    if (l2Cache) {
      // L1Cache hit, L2Cache hit
      return l2Cache;
    } else {
      // L1Cache hit, L2Cache does not hit
      //
      auto moduleRef = preprocessModuleOp(JitCode, NumArgs, TgtArgs, ArgPtrs,
                                          ArgSizes, ArgTypes);
      auto kernelFunc = lowerToStableHLO(moduleRef.get());
      return createL2JitMetas(
          l2CacheKey, kernelFunc,
          std::hash<std::string>{}(reinterpret_cast<char *>(JitCode)));
    }
  } else {
    // Neither L1Cache nor L2Cache hit.
    std::size_t kernelFingerprint =
        std::hash<std::string>{}(reinterpret_cast<char *>(JitCode));

    auto moduleRef = preprocessModuleOp(JitCode, NumArgs, TgtArgs, ArgPtrs,
                                        ArgSizes, ArgTypes);
    auto kernelFunc = lowerToStableHLO(moduleRef.get());

    auto shapeInfoMap = getShapeArgInfoMap(kernelFunc);
    auto l2Key = cacheManager.getL2JitMetasKey(NumArgs, TgtArgs, ArgSizes,
                                               JitCode, shapeInfoMap);
    cacheManager.saveL1JitMetas(JitCode, std::move(shapeInfoMap));
    return createL2JitMetas(l2Key, kernelFunc, kernelFingerprint);
  }
}

void JitManager::moveDataToHostBuffer(void *hostPtr, size_t size) {
  deviceManager.moveDataToHostBuffer(hostPtr, size);
}

void JitManager::destroyHostBoundBuffers(void *hostPtr) {
  deviceManager.destroyHostBoundBuffers(hostPtr);
}

bool JitManager::invalidateCachedDeviceBuffers(void *forgedTgtPtr,
                                               size_t size) {
  return deviceManager.invalidateDeviceBuffers(forgedTgtPtr, size);
}

// Coordinates of a device in the mesh; devices are numbered row-major (the
// last mesh axis varies fastest), as in Shardy's default device order.
// e.g. mesh {2, 3}: device 4 -> (1, 1).
static llvm::SmallVector<size_t>
getDevPosInMesh(size_t devIdx, llvm::ArrayRef<uint32_t> deviceMesh) {
  llvm::SmallVector<size_t> devPos(deviceMesh.size());
  for (int i = static_cast<int>(deviceMesh.size()) - 1; i >= 0; i--) {
    devPos[i] = devIdx % deviceMesh[i];
    devIdx /= deviceMesh[i];
  }
  return devPos;
}

// The shard a device at devPos holds: [start, end) on each dim. The axes of a
// dim sharding go from major to minor, so the shard index along that dim is
// mixed-radix over them, indexed by each axis's position in the mesh.
// e.g. mesh {a: 2, b: 3}, dim sharding {b, a}: shard = pos[b] * 2 + pos[a].
static ShardSlices getShardSlices(llvm::ArrayRef<int64_t> shape,
                                  const TensorSharding &sharding,
                                  llvm::ArrayRef<size_t> devPos) {
  assert(sharding.size() == shape.size() && "sharding rank mismatch");
  ShardSlices slices(shape.size());
  for (size_t dim = 0; dim < shape.size(); dim++) {
    size_t shardIdx = 0;
    size_t numShards = 1;
    for (const MeshAxis &axis : sharding[dim]) {
      shardIdx = shardIdx * axis.size + devPos[axis.ordinal];
      numShards *= axis.size;
    }
    size_t step = shape[dim] / numShards;
    slices[dim] = {shardIdx * step, (shardIdx + 1) * step};
  }
  return slices;
}

void JitManager::prepareInputBuffers(
    const TensorDesc *args, const unsigned int count,
    std::vector<std::vector<PJRT_Buffer *>> &inBuffers,
    const TensorShardings &sd) {
  // TODO: be compatible with existing plugins.
  // currently only consider TPU runtime plugin

  // TODO: this should be parallelized
  for (size_t devIdx = 0; devIdx < sharder.deviceCount; devIdx++) {
    auto devPos = getDevPosInMesh(devIdx, sharder.deviceMesh);
    for (unsigned argIdx = 0; argIdx < count; argIdx++) {
      const TensorDesc &arg = args[argIdx];
      if (arg.rank == 0) {
        inBuffers[devIdx][argIdx] =
            deviceManager.createLiteralBuffer(devIdx, arg);
        continue;
      }
      ShardSlices slices = getShardSlices(arg.shape, sd[argIdx], devPos);
      inBuffers[devIdx][argIdx] = deviceManager.moveDataSegsToDevice(
          devIdx, arg.data, arg.shape, slices, arg.dtype);
    }
  }
}

void JitManager::executeOnMultiDevices(PJRT_LoadedExecutable *exe,
                                       PJRT_Buffer ***argLists,
                                       PJRT_Buffer **const *outLists,
                                       const int in_args_count) {
  PJRT_ExecuteOptions execute_options = {
      .struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
  };
  const int deviceCount = this->deviceManager.pjrtDevices_.size();
  llvm::SmallVector<PJRT_Event *> deviceCompleteEvents(deviceCount);

  PJRT_LoadedExecutable_Execute_Args leeas = {
      .struct_size =
          PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE, // function and args
      .executable = exe,
      .options = &execute_options,
      .argument_lists = argLists, // [deviceCount][argCount],
      .num_devices = (size_t)deviceCount,
      .num_args = (size_t)in_args_count,
      .output_lists = outLists,
      .device_complete_events = deviceCompleteEvents.data(),
      .execute_device = nullptr,
  };

  auto executeErr = pjrtApi->PJRT_LoadedExecutable_Execute(&leeas);
  if (executeErr) {
    std::cerr << "[Error] Execute LoadedExecutable: "
              << JitManager::getErrMsg(pjrtApi, executeErr) << "\n";
    std::exit(EXIT_FAILURE);
  }

  for (int i = 0; i < deviceCount; i++) {
    if (leeas.device_complete_events != nullptr &&
        leeas.device_complete_events[i] != nullptr) {
      PJRT_Event_Await_Args waitArgs = {
          .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
          .event = leeas.device_complete_events[i]};
      pjrtApi->PJRT_Event_Await(&waitArgs);
      PJRT_Event_Destroy_Args eda = {.struct_size =
                                         PJRT_Event_Destroy_Args_STRUCT_SIZE,
                                     .event = leeas.device_complete_events[i]};
      pjrtApi->PJRT_Event_Destroy(&eda);
    }
  }
}

// TODO: this only works when using the TPU plugin, where all the memory
void JitManager::manageOutBuffers(int inArgsCount,
                                  PJRT_Buffer **const *outsBuffersList,
                                  TensorDesc *inputArgs,
                                  TensorDesc *outputArgs,
                                  const TensorShardings &outputShardings) {

  for (int devIdx = 0; devIdx < this->deviceManager.pjrtDevices_.size();
       devIdx++) {
    auto devPos = getDevPosInMesh(devIdx, sharder.deviceMesh);
    for (int argIdx = 0; argIdx < inArgsCount; argIdx++) {
      auto inputArg = inputArgs[argIdx];
      if (inputArg.isLiteral) {
        deviceManager.destroyPJRTBuffer(this->pjrtApi,
                                        outsBuffersList[devIdx][argIdx]);
        continue;
      }
      // Result i is written back to argument i, so it has argument i's shape.
      ShardSlices slices =
          getShardSlices(inputArg.shape, outputShardings[argIdx], devPos);
      deviceManager.moveOutBuffersToDeviceBufferMap(
          devIdx, inputArg.data, argIdx, outsBuffersList, inputArg.shape,
          slices, getDTypeSizeInByte(inputArg.dtype));
    }
  }
}

void JitManager::launch(const L2JitMetas* l2Cache, KernelArgs *kernelArgs) {
  auto idAndStartPair = sharder.startRecordProfile(l2Cache->ioShardings.inputs, l2Cache->kernelFingerprint, *kernelArgs);
  DEBUG_PRINT("Enter launchKernelOnMultiDevices");
  // TODO: improve this when it becomes slow
  // InputBufs[deviceId][argIdx]
  std::vector<std::vector<PJRT_Buffer *>> inputBufs(
      deviceManager.pjrtDevices_.size(),
      std::vector<PJRT_Buffer *>(kernelArgs->inputArgCount));

  // Move data from input data `offloading -> inputArgs` to `inputBufs`.
  // InputBufs are buffers in each devices.
  DEBUG_PRINT("Before manage input buffers");
  prepareInputBuffers(kernelArgs->inputArgs, kernelArgs->inputArgCount,
                      inputBufs, l2Cache->ioShardings.inputs);
  DEBUG_PRINT("Finish manage input buffers");

  PJRT_Buffer ***inputBufsList;
  std::vector<PJRT_Buffer **> rawInputBufs(inputBufs.size());
  for (int i = 0; i < inputBufs.size(); i++) {
    rawInputBufs[i] = inputBufs[i].data();
  }
  inputBufsList = rawInputBufs.data();

  std::vector<std::vector<PJRT_Buffer *>> outputArgsBufs(
      deviceManager.pjrtDevices_.size(),
      std::vector<PJRT_Buffer *>(kernelArgs->outputArgCount));

  // outputBufsList stores the data handler for each output on each device.
  // [device][argIdx]
  PJRT_Buffer **const *outputBufsList;
  std::vector<PJRT_Buffer **> rawOutputBufs(outputArgsBufs.size());
  for (int i = 0; i < outputArgsBufs.size(); i++) {
    rawOutputBufs[i] = outputArgsBufs[i].data();
  }
  outputBufsList = rawOutputBufs.data();

  DEBUG_PRINT("Before executing");
  executeOnMultiDevices(l2Cache->exe, inputBufsList, outputBufsList,
                        kernelArgs->inputArgCount);
  DEBUG_PRINT("Finish executing");

  // move data back to the host
  manageOutBuffers(kernelArgs->inputArgCount, outputBufsList,
                   kernelArgs->inputArgs, kernelArgs->outputArgs,
                   l2Cache->ioShardings.outputs);
  DEBUG_PRINT("Finish manage out buffers");
  sharder.endRecordProfile(idAndStartPair.first, std::move(idAndStartPair.second));
  return;
}


void JitManager::tune() {
  llvm::DenseMap<size_t, func::FuncOp> map;
  llvm::SmallVector<size_t> series;
  this->tuner.tune(this->context, map, series, sharder.meshAxes);
};
