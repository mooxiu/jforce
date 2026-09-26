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
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <unordered_map>
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
      deviceManager(pjrtApi, pjrtClient, targetDeviceTy) {
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
  mlir::OpBuilder opBuilder(&this->context);
  // TODO: delete this after testing!
  kernelFunc = hardcodedShard(kernelFunc, opBuilder);
  DEBUG_PRINT_OP(kernelFunc);
  auto moduleOp = kernelFunc->getParentOfType<mlir::ModuleOp>();
  auto kernelFuncStr = getMLIROperationAsString(moduleOp);
  auto exec = this->compilePJRTExecutable(kernelFuncStr);
  return cacheManager.insertL2CacheAndReturn(
      key, exec, std::move(kernelFuncStr), kernelFunc,
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

void JitManager::launch(PJRT_LoadedExecutable *exec, KernelArgs *kernelArgs,
                        const std::string &kernelFuncStr) {
  deviceManager.launchKernelOnMultiDevices(exec, kernelArgs, kernelFuncStr);
}
