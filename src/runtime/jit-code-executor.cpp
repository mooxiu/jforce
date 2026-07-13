#include "../support/profiler.h"
#include "../support/utilities.h"
#include "../transform/Passes.h"
#include "jit-manager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "transform/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

using namespace mlir;

#define JIT_LITERAL_VAL_ATTR_NAME "jit.literal_val"
#define JIT_ARGS_MAPPING_ATTR_NAME "jit.args_mapping"


static TargetDevice getTargetDevice() {
  static const TargetDevice device = [] {
    const char *value = std::getenv("JFORCE_TARGET_DEVICE");

    if (!value || *value == '\0') {
      llvm::report_fatal_error(
          "JFORCE_TARGET_DEVICE is not set; "
          "expected CPU, CUDA, ROCM, or TPU.");
    }

    std::string name(value);
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::toupper(c));
                   });

    if (name == "CPU")
      return TargetDevice::CPU;
    if (name == "CUDA")
      return TargetDevice::CUDA;
    if (name == "ROCM")
      return TargetDevice::ROCM;
    if (name == "TPU")
      return TargetDevice::TPU;

    llvm::errs() << "Unsupported device!\n";
    exit(EXIT_FAILURE);
  }();

  return device;
}

#ifdef ENABLE_XLA_DEBUG
#define PRINT_PASS() \
  ctx->disableMultithreading(); pm.enableIRPrinting()
#else
#define PRINT_PASS()
#endif

bool fillKernelFuncArgs(llvm::DenseMap<unsigned, unsigned> argsIndicesMapping,
                        TensorDesc *newArgs, ArrayRef<Type> kernelFuncTypes,
                        int64_t NumHostArgs, int64_t *ArgTypes,
                        void **TgtArgs) {
  for (int oldIdx = 0; oldIdx < NumHostArgs; oldIdx++) {
    if (!argsIndicesMapping.contains(oldIdx)) {
      continue;
    }
    auto newIdx = argsIndicesMapping.at(oldIdx);
    auto thisTy = kernelFuncTypes[newIdx];
    assert(llvm::isa<RankedTensorType>(thisTy) && "Suppose all args are ");
    auto rtType = llvm::dyn_cast<RankedTensorType>(thisTy);

    newArgs[newIdx] = TensorDesc{
        .data = TgtArgs[oldIdx],
        .shape = rtType.getShape().data(),
        .rank = (int32_t)rtType.getRank(),
        .dtype = getDTypeFromRankedTensorType(rtType),
        .isLiteral = isLiteralTy(ArgTypes[oldIdx]),
    };
  }
  return true;
}

struct jitArg {
  void *hostPtr;
  void *tgtPtr;
  int64_t size;
  bool isLiteral;
};

llvm::SmallVector<jitArg> packJitArg(int64_t NumArgs, void **TgtArgs,
                                     void **ArgPtrs, int64_t *ArgSizes,
                                     int64_t *ArgTypes) {
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
}

void insertJitInfo(mlir::OpBuilder& builder, func::FuncOp kernelFunc, llvm::SmallVector<jitArg> args) {
  auto ctx = builder.getContext();
  auto attrName = builder.getStringAttr(JIT_LITERAL_VAL_ATTR_NAME);
  for (int i = 0; i < args.size(); i++) {
    if (args[i].isLiteral) {
      std::uintptr_t literalAddr = reinterpret_cast<std::uintptr_t>(args[i].hostPtr);
      auto attr = IntegerAttr::get(
        IntegerType::get(ctx, sizeof(void*) * 8),
        literalAddr
      );
      kernelFunc.setArgAttr(i, attrName, attr);
    }
  }
  return;
}

llvm::DenseMap<uint32_t, uint32_t> rebuildIndicesMapping(func::FuncOp funcOp) {
  llvm::DenseMap<uint32_t, uint32_t> indicesMap;
  auto arrayAttr = funcOp->getAttrOfType<ArrayAttr>(JIT_ARGS_MAPPING_ATTR_NAME);
  assert(arrayAttr && "Jit args map after trimming should be stored as attribute!");
  auto attrs = arrayAttr.getValue();
  for (int i = 0; i < attrs.size(); i+=2) {
    uint32_t key = llvm::cast<IntegerAttr>(attrs[i]).getUInt();
    uint32_t val = llvm::cast<IntegerAttr>(attrs[i+1]).getUInt();
    indicesMap[key] = val;
  }
  return indicesMap;
}

// ------------------------------ Init ------------------------------
/**
 * JitCode: A function contains the omp::TargetOp with a omp::workdistributeOp
 * inside.
 *
 */
extern "C" int64_t __botw_jit_code(void *JitCode, int64_t NumArgs,
                                   void **TgtArgs, ptrdiff_t *TgtOffsets,
                                   void *DeviceArgs, int64_t NumHostArgs,
                                   void **ArgBasePtrs, void **ArgPtrs,
                                   int64_t *ArgSizes, int64_t *ArgTypes,
                                   void **ArgNames) {
  PROFILE_SCOPE("total", Phase::TOTAL);
  char *JitCodeC = reinterpret_cast<char *>(JitCode);
  std::cerr << "Got a jit call with " << NumArgs << " args into:\n" <<
  JitCodeC << "\n"; llvm::dbgs() << "\nreceive a jit call\n";

  #define p(A) std::cerr << " " << #A << ": " << A[I] << "\n"
  #define h(A) \
    std::cerr << " " << #A << std::hex << ": 0x" << A[I] << std::dec << "\n"
    for (unsigned I = 0; I < NumArgs; I++) {
      std::cerr << "Device Arg #" << I << ":\n";
      p(TgtArgs);
      p(TgtOffsets);
    }
    for (unsigned I = 0; I < NumHostArgs; I++) {
      std::cerr << "Host Arg #" << I << ":\n";
      p(ArgBasePtrs);
      p(ArgPtrs);
      p(ArgSizes);
      h(ArgTypes);
      h(ArgNames);
    }
  #undef p
  #undef h

  assert(NumArgs == NumHostArgs);
  // there is an extra pointer added...
  // TODO: Temporary only
  NumArgs -= 1;
  NumHostArgs -= 1;
  ArgSizes -= 1;
  

  auto JitCodePtrUint = reinterpret_cast<uintptr_t>(JitCode);
  auto l1JitMetas = JitManager::getInstance().tryGetL1JitMetas(JitCodePtrUint);
  llvm::SmallVector<uint64_t, 128> l2Key;

  if (l1JitMetas != nullptr) {
    l2Key = JitManager::getInstance().getL2JitMetasKey(
        NumArgs, ArgTypes, TgtArgs, ArgSizes, JitCodePtrUint,
        l1JitMetas->argsIndices);
    auto l2JitMetas = JitManager::getInstance().tryGetL2JitMetas(l2Key);

    if (l2JitMetas != nullptr) {
      // auto kernelFunc = jitMeta->kernelFunc;
      std::vector<TensorDesc> newArgs(l2JitMetas->kernelFuncTypes.size());
      fillKernelFuncArgs(l2JitMetas->argsIndicesMapping, newArgs.data(),
                         l2JitMetas->kernelFuncTypes, NumHostArgs, ArgTypes,
                         TgtArgs);
      auto kArgs =
          (KernelArgs){.inputArgCount = l2JitMetas->argsIndicesMapping.size(),
                       .inputArgs = newArgs.data(),
                       .outputArgCount = l2JitMetas->argsIndicesMapping.size(),
                       .outputArgs = newArgs.data(),
                       .targetDevice = getTargetDevice()};
      JitManager::getInstance().launchKernel(
          l2JitMetas->exe, &kArgs, JitCodePtrUint, l2JitMetas->kernelFuncStr);
      return 0;
    }
  }

  // Parse JitCode to ModuleOp
  MLIRContext *ctx = JitManager::getInstance().getContext();
  // Use OweningOpRef so RAII can help to destroy the tree
  mlir::OwningOpRef<mlir::ModuleOp> moduleOpRef = JitManager::getInstance().getModuleOp(JitCodePtrUint, JitCodeC);
  mlir::OpBuilder builder(ctx);
  auto args = packJitArg(NumArgs, TgtArgs, ArgPtrs, ArgSizes, ArgTypes);
  auto moduleOp = moduleOpRef.get(); 
  auto kernel = moduleOp.lookupSymbol<func::FuncOp>("kernel");
  assert(kernel && "FuncOp with name kernel should exist!");
  insertJitInfo(builder, kernel, args);

  mlir::PassManager pm(ctx);
  // PRINT_PASS();
  // pm.enableCrashReproducerGeneration("./crash_repro.mlir");
  // pm.enableTiming();

  // 1. AOT -> Shape Inference
  auto& nestedPMPhase1 = pm.nest<mlir::func::FuncOp>();
  nestedPMPhase1.addPass(xla_jit::createCleanTempsPass());
  nestedPMPhase1.addPass(createCanonicalizerPass());
  nestedPMPhase1.addPass(xla_jit::createAnnotatePass());
  nestedPMPhase1.addPass(xla_jit::createPropagateConstantsPass());
  nestedPMPhase1.addPass(createCanonicalizerPass());
  nestedPMPhase1.addPass(createSCCPPass());
  nestedPMPhase1.addPass(createCSEPass());
  nestedPMPhase1.addPass(createCanonicalizerPass());
  nestedPMPhase1.addPass(xla_jit::createShapeInferPass());
  nestedPMPhase1.addPass(createCanonicalizerPass());

  // 2. ShapeInference HLFIR -> FIR -> Optimize -> Memref, Affine, SCF
  // - 2.1 HLFIR -> FIR
  pm.addPass(hlfir::createConvertHLFIRtoFIR());
  // - 2.2 FIR and optimize
  auto& nestedPMPhase2 = pm.nest<mlir::func::FuncOp>();
  nestedPMPhase2.addPass(xla_jit::createFoldRepeatConversionsPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(createLoopInvariantCodeMotionPass());
  nestedPMPhase2.addPass(createCSEPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(xla_jit::createMemOpsFoldingPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(xla_jit::createCleanFIRLoopPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  // - 2.3 FIR -> MemRef
  nestedPMPhase2.addPass(fir::createFIRToMemRef());
  nestedPMPhase2.addPass(xla_jit::createCleanFIROpsPass());
  nestedPMPhase2.addPass(fir::createPromoteToAffinePass());
  nestedPMPhase2.addPass(affine::createAffineLoopNormalizePass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(fir::createFIRToSCFPass()); // for fir.if -> scf.if
  nestedPMPhase2.addPass(createCanonicalizerPass());

  // 3. Common MLIR -> Affine Loops
  nestedPMPhase2.addPass(xla_jit::createFoldSCFIfPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(xla_jit::createPolygeistMem2RegPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(xla_jit::createLoopSinkingPass());
  nestedPMPhase2.addPass(xla_jit::createRecognizeMinMaxPass());
  // pm.addPass(xla_jit::createAffineCFGPass());


  // 4. Outline Affine loops, Tensorize, mergeback
  pm.addPass(xla_jit::createOutlineAffinePass());
  pm.addPass(xla_jit::createAffineCFGPass());
  auto& nestedPMPhase3 = pm.nest<mlir::func::FuncOp>();
  nestedPMPhase3.addPass(xla_jit::createAffineToStableHLORaisingPass());
  nestedPMPhase3.addPass(xla_jit::createArithRaisingPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(xla_jit::createRemergePass());
  pm.addPass(xla_jit::createWorkdistributeToStableHLOPass());
  pm.addPass(createInlinerPass());
  auto& nestedPMPhase4 = pm.nest<mlir::func::FuncOp>();
  nestedPMPhase4.addPass(stablehlo::createStablehloAggressiveSimplificationPass());
  nestedPMPhase4.addPass(xla_jit::createAliasingPass());
  nestedPMPhase4.addPass(xla_jit::createTrimArgsPass());
  if (mlir::failed(pm.run(moduleOp))) {
    llvm::errs() << "MLIR Pass Pipeline failed!\n";
    std::exit(EXIT_FAILURE);
  }

  // kernel function needs to be named as `main` to be compiled by XLA
  auto kernelFunc = moduleOp.lookupSymbol<func::FuncOp>("main");
  assert(kernelFunc && "Kernel Func should be renamed as main!\n");
  auto argsIndicesMapping = rebuildIndicesMapping(kernelFunc);

  if (l1JitMetas == nullptr) {
    llvm::DenseSet<int> argsIndices;
    for (int i = 0; i < NumArgs; i++) {
      // not contains in argsIndicesMapping, meaning it's the shape arguments
      // that been trimmed above
      if (!argsIndicesMapping.contains(i)) {
        argsIndices.insert(i);
      }
    }
    l2Key = JitManager::getInstance().getL2JitMetasKey(NumArgs, ArgTypes, TgtArgs, ArgSizes, JitCodePtrUint, argsIndices);
    JitManager::getInstance().saveL1JitMetas(JitCodePtrUint, std::move(argsIndices));
  }

  auto createdL2JitMetas = JitManager::getInstance().createL2JitMetas(
      l2Key, kernelFunc, argsIndicesMapping, getTargetDevice());

  std::vector<TensorDesc> newArgs(createdL2JitMetas->kernelFuncTypes.size());

  if (fillKernelFuncArgs(argsIndicesMapping, newArgs.data(),
                         createdL2JitMetas->kernelFuncTypes, NumHostArgs,
                         ArgTypes, TgtArgs)) {
    KernelArgs args = (struct KernelArgs){
        .inputArgCount = argsIndicesMapping.size(),
        .inputArgs = newArgs.data(),
        .outputArgCount = argsIndicesMapping.size(),
        .outputArgs = newArgs.data(),
        .targetDevice = getTargetDevice(),
    };
    JitManager::getInstance().launchKernel(createdL2JitMetas->exe, &args,
                                           JitCodePtrUint,
                                           createdL2JitMetas->kernelFuncStr);
  } else {
    std::cerr << "Fail to fill kernel func args!\n";
    return 1;
  }

  return 0;
}
