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
#include "pipelines.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
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
  ctx->disableMultithreading();                                                \
  pm.enableIRPrinting()
#else
#define PRINT_PASS()
#endif

// FIXME: used for debugging only, delete this after usage
static void printBufferShape(const PJRT_Api *api, PJRT_Buffer *buffer,
                             const char *label, int device, int arg) {
  PJRT_Buffer_Dimensions_Args q{};
  q.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  q.buffer = buffer;
  if (auto *err = api->PJRT_Buffer_Dimensions(&q)) {
    llvm::errs() << JitManager::getErrMsg(api, err) << "\n";
    return;
  }
  llvm::errs() << label << " device=" << device << " arg=" << arg << " [";
  for (size_t d = 0; d < q.num_dims; ++d) {
    llvm::errs() << (d ? "," : "") << q.dims[d];
    llvm::errs() << "]\n";
  }
}

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

JitManager::JitManager()
    : pjrtApi(loadPjrtAPi()), pjrtClient(getPJRTClient(pjrtApi)),
      targetDeviceTy(getTargetDeviceFromEnv()), cacheManager(context),
      deviceManager(pjrtApi, pjrtClient, targetDeviceTy),
      sharder(deviceManager.pjrtDevices_.size()) {
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

mlir::func::FuncOp JitManager::lowerToStableHLO(mlir::ModuleOp moduleOp) {
  // Lower JItCode to StableHLO
  mlir::PassManager pm(&this->context);
  // INFO: uncomment print pass when we need to debug the IR transformation
  // PRINT_PASS();
  pm.enableCrashReproducerGeneration("./crash_repro.mlir");
  // pm.enableTiming();
  createLowerToStableHLOPassPipeline(pm);
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

mlir::ModuleOp JitManager::annoateShardyInfo(mlir::ModuleOp moduleOp,
                                             const ShardingDecision &sd) {
  OpBuilder opBuilder(&context);
  context.getOrLoadDialect<mlir::sdy::SdyDialect>();

  auto kernelFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>("main");
  assert(kernelFunc && "There should exist a kernel function called main");

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

  // Sharding strategy on each argument
  assert(sd.size() == kernelFunc.getNumArguments() &&
         "Sharding Decision does not have the same size as the function "
         "arguments count.");
  llvm::SmallVector<Attribute> attrs(kernelFunc.getNumArguments(),
                                     opBuilder.getDictionaryAttr({}));
  for (unsigned int i = 0; i < kernelFunc.getNumArguments(); i++) {
    llvm::SmallVector<sdy::DimensionShardingAttr> dimShardings;
    dimShardings.reserve(sd[i].size());
    for (const auto &tensorDimSharding : sd[i]) {
      llvm::SmallVector<sdy::AxisRefAttr> axes;
      for (const auto &axis : tensorDimSharding) {
        axes.push_back(sdy::AxisRefAttr::get(&context, axis.get().name));
      }
      auto dimShardingAttr =
          sdy::DimensionShardingAttr::get(&context, axes, true); // is closed
      dimShardings.push_back(dimShardingAttr);
    }

    attrs[i] = opBuilder.getDictionaryAttr(opBuilder.getNamedAttr(
        mlir::sdy::TensorShardingAttr::name,
        mlir::sdy::TensorShardingAttr::get(
            &context, sharder.meshName,
            /*dim_shardings=*/dimShardings,
            /*replicated_axes=*/{}, // WARNING: how to deal with this?
            /*unreduced_axes=*/{}   // WARANING: and this?
            )));
  }
  kernelFunc.setArgAttrsAttr(opBuilder.getArrayAttr(attrs));
  return moduleOp;
}

L2JitMetas *
JitManager::createL2JitMetas(const llvm::SmallVector<uint64_t, 128> &key,
                             mlir::func::FuncOp kernelFunc) {
  mlir::OpBuilder opBuilder(&this->context);
  auto moduleOp = kernelFunc->getParentOfType<mlir::ModuleOp>();
  ShardingDecision sd = sharder.heuristicShard(moduleOp);
  moduleOp = annoateShardyInfo(moduleOp, sd);
  auto kernelFuncStr = getMLIROperationAsString(moduleOp);
  auto exec = this->compilePJRTExecutable(kernelFuncStr);

  auto funcTypes = kernelFunc.getFunctionType().getInputs();
  std::vector<mlir::Type> argTypesVec(funcTypes.begin(), funcTypes.end());
  L2JitMetas l2Cache(exec, std::move(argTypesVec), std::move(kernelFuncStr),
                     std::move(sd));
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
      return createL2JitMetas(l2CacheKey, kernelFunc);
    }
  } else {
    // Neither L1Cache nor L2Cache hit.
    auto moduleRef = preprocessModuleOp(JitCode, NumArgs, TgtArgs, ArgPtrs,
                                        ArgSizes, ArgTypes);
    auto kernelFunc = lowerToStableHLO(moduleRef.get());

    auto shapeInfoMap = getShapeArgInfoMap(kernelFunc);
    auto l2Key = cacheManager.getL2JitMetasKey(NumArgs, TgtArgs, ArgSizes,
                                               JitCode, shapeInfoMap);
    cacheManager.saveL1JitMetas(JitCode, std::move(shapeInfoMap));
    return createL2JitMetas(l2Key, kernelFunc);
  }
}

void JitManager::moveDataToHostBuffer(void *hostPtr, size_t size) {
  deviceManager.moveDataToHostBuffer(hostPtr, size);
}

void JitManager::destroyHostBoundBuffers(void *hostPtr) {
  deviceManager.destroyHostBoundBuffers(hostPtr);
}

bool JitManager::invalidateCachedDeviceBuffers(void *forgedTgtPtr, size_t size) {
  return deviceManager.invalidateDeviceBuffers(forgedTgtPtr, size);
}

// TODO: for each device, we probably should use thread to do the IO
static void prepareInputBuffersForSingleDevice(
    int devIdx, DeviceManager &devManager, const TensorDesc *args,
    const unsigned int count, std::vector<std::vector<PJRT_Buffer *>>& inBuffers,
    const ShardingDecision &sd, const llvm::SmallVector<size_t> &devPos) {

  for (int argIdx = 0; argIdx < count; argIdx++) {
    auto arg = args[argIdx];
    auto argSD = sd[argIdx];

    if (arg.rank == 0) {
      inBuffers[devIdx][argIdx] = devManager.createLiteralBuffer(devIdx, arg);
      continue;
    }

    // slices of the tensor on each dimension for this argument tensor
    llvm::SmallVector<std::pair<size_t, size_t>> tensorSlices(arg.rank);
    for (int argDimIdx = 0; argDimIdx < arg.rank; argDimIdx++) {
      const auto &dimSD = argSD[argDimIdx];
      if (dimSD.empty()) {
        // not sharding this dimension
        tensorSlices[argDimIdx] = std::make_pair(0, arg.shape[argDimIdx]);
      } else {
        auto prod = 1;
        auto offset = 0;
        auto multiplier = 1;
        // loop for right to left for computing offset at the same time
        for (int i = dimSD.size() - 1; i >= 0; i--) {
          const auto &meshAxis = dimSD[i];
          prod *= meshAxis.get().size;
          if (i == dimSD.size() - 1) {
            multiplier = 1;
          } else {
            multiplier = multiplier * dimSD[i + 1].get().size;
          }
          offset += devPos[i] * multiplier;
        }
        assert(prod > 0 && "sharding prod should be greater than 0.");
        auto sliceStep = arg.shape[argDimIdx] / prod;

        // in most cases, we only have 1 or 2 dims, no need to cache the prefix
        // prod
        auto start = offset * sliceStep;
        tensorSlices[argDimIdx] = std::make_pair(start, start + sliceStep);
      }
    }
    inBuffers[devIdx][argIdx] = devManager.moveDataSegsToDevice(
        devIdx, arg.data, arg.shape, tensorSlices, arg.dtype);

    /**
     * No need to map to 1-dimensional memory as API can help us.
     *
    // We have to mapping from a tensor's representation: T(s_0, s_1, ...
    // s_{n-1}) to 1 dimensional memory, Each s_i is a continuous slice,
    // represented as a pair of [start, end). For an n-dimensional tensor like
    // T, it will correspond to $\prod_{i=0}^{n-2}(s[i][1] - s[i][0])$ segments.
    // This is likely a horrifying number but in most cases we only have
    // 2-dimensional tensors. Furthermore, we have an optimization: if the slice
    // can be something like T(s_0, s_1, ..., s_k, :,..., :), The memory after
    // dim-k is continuous.
    //
    // memSegs are memory segements start with void* and length uint32_t
    llvm::SmallVector<std::pair<void *, size_t>> memSegs;
    int pivot = -1;
    // find pivot dim where arg.shape[pivot] > tensorSlices[pivot]
    for (int i = arg.rank - 1; i >= 0; i--) {
      const auto sliceSize = tensorSlices[i].second - tensorSlices[i].first;
      if (arg.shape[i] != sliceSize) {
        pivot = i;
        break;
      }
    }

    if (pivot != -1) {
      // This works like a prefix production.
      // For example, if we have 2-dimensional matrix of [6, 4], with float64
      // Then byteStrides is [32, 8]
      // Becuse now, any change in dim0 will bring size(dim1) * size(float64)
      // bytes changes, and change in dim1 will bring size(float64) changes.
      llvm::SmallVector<size_t> byteStrides(arg.rank);
      size_t stride = arg.getDTypeSize();
      for (int i = arg.rank - 1; i >= 0; i--) {
        byteStrides[i] = stride;
        stride *= static_cast<size_t>(arg.shape[i]);
      }
      // granularity of continuous memory
      const size_t segBytes =
          (tensorSlices[pivot].second - tensorSlices[pivot].first) *
          byteStrides[pivot];

      llvm::SmallVector<size_t> coords(pivot);
      for (int i = 0; i < pivot; i++) {
        coords[i] = tensorSlices[i].first;
      }

      auto *base = static_cast<char *>(arg.data);
      while (true) {
        size_t byteOffset = tensorSlices[pivot].first * byteStrides[pivot];
        for (int i = 0; i < pivot; i++) {
          byteOffset += coords[i] * byteStrides[pivot];
        }

        memSegs.emplace_back(base + byteOffset, segBytes);

        // advance in coords, if exceed slices.second, carry
        int i = pivot - 1;
        for (; i >= 0; i--) {
          coords[i] += 1;
          if (coords[i] < tensorSlices[i].second) {
            // no need to carry
            break;
          }
          coords[i] = tensorSlices[i].first;
        }
        if (i < 0) {
          break;
        }
      }
    } else {
      // There is no data sharding at all, just move the whole segment
      assert(pivot == -1);
      memSegs.push_back(
          std::make_pair(arg.data, arg.getEleCount() * arg.getDTypeSize()));
    }
    deviceManager.moveDataToDevice(std::move(memSegs));
   */
  }
}

void JitManager::prepareInputBuffers(
    const TensorDesc *args, const unsigned int count,
    std::vector<std::vector<PJRT_Buffer *>>& inBuffers,
    const ShardingDecision &sd) {
  // TODO: be compatible with existing plugins.
  // currently only consider TPU runtime plugin

  llvm::SmallVector<uint32_t> meshPrefixProd(sharder.deviceMesh.size());
  for (int i = sharder.deviceMesh.size() - 1; i >= 0; i--) {
    if (i == sharder.deviceMesh.size() - 1) {
      meshPrefixProd[i] = sharder.deviceMesh[i];
    } else {
      meshPrefixProd[i] *= sharder.deviceMesh[i];
    }
  }
  auto getDevPosInMesh = [&](size_t devIdx) {
    llvm::SmallVector<size_t> devPos(sharder.deviceMesh.size());
    size_t offset = 0;
    for (int i = 0; i < sharder.deviceMesh.size(); i++) {
      if (i < sharder.deviceMesh.size() - 1) {
        devPos[i] = (devIdx - offset) / meshPrefixProd[i + 1];
        offset += devPos[i] * meshPrefixProd[i + 1];
      } else {
        devPos[i] = devIdx - offset;
      }
    }
    return devPos;
  };

  // TODO: this should be parallelized
  for (size_t devIdx = 0; devIdx < sharder.deviceCount; devIdx++) {
    auto devPos = getDevPosInMesh(devIdx);
    prepareInputBuffersForSingleDevice(devIdx, deviceManager, args, count,
                                       inBuffers, sd, devPos);
  }
  return;
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
                                  TensorDesc *outputArgs) {

  for (int devIdx = 0; devIdx < this->deviceManager.pjrtDevices_.size();
       devIdx++) {
    for (int argIdx = 0; argIdx < inArgsCount; argIdx++) {
      auto inputArg = inputArgs[argIdx];
      if (inputArg.isLiteral) {
        deviceManager.destroyPJRTBuffer(this->pjrtApi,
                                        outsBuffersList[devIdx][argIdx]);
        continue;
      }
      deviceManager.moveOutBuffersToDeviceBufferMap(devIdx, inputArg.data,
                                                    argIdx, outsBuffersList);
      // FIXME: delete after debugging
      printBufferShape(pjrtApi, outsBuffersList[devIdx][argIdx], "[DEBUG OUT]",
                       devIdx, argIdx);
    }
  }
}

void JitManager::launch(PJRT_LoadedExecutable *exec, KernelArgs *kernelArgs,
                        const std::string &kernelFuncStr,
                        const ShardingDecision &sd) {
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
                      inputBufs, sd);
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
  executeOnMultiDevices(exec, inputBufsList, outputBufsList,
                        kernelArgs->inputArgCount);
  DEBUG_PRINT("Finish executing");

  // move data back to the host
  manageOutBuffers(kernelArgs->inputArgCount, outputBufsList,
                   kernelArgs->inputArgs, kernelArgs->outputArgs);
  DEBUG_PRINT("Finish manage out buffers");
  return;
}
