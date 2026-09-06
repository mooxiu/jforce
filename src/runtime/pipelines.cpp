#include "pipelines.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "../transform/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

void createLowerToFixedShapedIRPipeline(PassManager& pm) {
  auto &nestedPMPhase1 = pm.nest<mlir::func::FuncOp>();
  // nestedPMPhase1.addPass(xla_jit::createCleanTempsPass());
  nestedPMPhase1.addPass(createCanonicalizerPass());
  nestedPMPhase1.addPass(xla_jit::createAnnotatePass());
  nestedPMPhase1.addPass(xla_jit::createPropagateConstantsPass());
  nestedPMPhase1.addPass(createCanonicalizerPass());
  nestedPMPhase1.addPass(createSCCPPass());
  nestedPMPhase1.addPass(createCSEPass());
  nestedPMPhase1.addPass(createCanonicalizerPass());
  nestedPMPhase1.addPass(xla_jit::createShapeInferPass());
  nestedPMPhase1.addPass(createCanonicalizerPass());
} 

void createLowerToAffineIRPipeline(PassManager& pm) {
  pm.addPass(hlfir::createConvertHLFIRtoFIR());
  auto &nestedPMPhase2 = pm.nest<mlir::func::FuncOp>();
  // To FIR
  nestedPMPhase2.addPass(xla_jit::createFoldRepeatConversionsPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(createLoopInvariantCodeMotionPass());
  nestedPMPhase2.addPass(createCSEPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(xla_jit::createCleanFIRLoopPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(xla_jit::createMemOpsFoldingPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());

  // To General used IR
  nestedPMPhase2.addPass(fir::createFIRToMemRef());
  nestedPMPhase2.addPass(xla_jit::createCleanFIROpsPass());
  nestedPMPhase2.addPass(fir::createPromoteToAffinePass());
  nestedPMPhase2.addPass(affine::createAffineLoopNormalizePass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(fir::createFIRToSCFPass()); // for fir.if -> scf.if
  nestedPMPhase2.addPass(createCanonicalizerPass());

  // To Affine IR
  nestedPMPhase2.addPass(xla_jit::createFoldSCFIfPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(xla_jit::createPolygeistMem2RegPass());
  nestedPMPhase2.addPass(createCanonicalizerPass());
  nestedPMPhase2.addPass(xla_jit::createLoopSinkingPass());
  nestedPMPhase2.addPass(xla_jit::createRecognizeMinMaxPass());
  // pm.addPass(xla_jit::createAffineCFGPass());
} 

void createTranslateToStableHLOPipeline(PassManager& pm) {
  // 4. Outline Affine loops, Tensorize, mergeback
  pm.addPass(xla_jit::createOutlineAffinePass());
  pm.addPass(xla_jit::createAffineCFGPass());
  auto &nestedPMPhase3 = pm.nest<mlir::func::FuncOp>();
  nestedPMPhase3.addPass(xla_jit::createAffineToStableHLORaisingPass());
  nestedPMPhase3.addPass(xla_jit::createArithRaisingPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(xla_jit::createRemergePass());
  // pm.addPass(xla_jit::createWorkdistributeToStableHLOPass());
  pm.addPass(xla_jit::createTranslatePass());
  pm.addPass(createInlinerPass());
}

void createLowerToStableHLOPassPipeline(PassManager& pm) {
  createLowerToFixedShapedIRPipeline(pm);
  createLowerToAffineIRPipeline(pm);
  createTranslateToStableHLOPipeline(pm);

  auto &nestedPMPhase4 = pm.nest<mlir::func::FuncOp>();
  nestedPMPhase4.addPass(
      stablehlo::createStablehloAggressiveSimplificationPass());
  nestedPMPhase4.addPass(xla_jit::createAliasingPass());
  // nestedPMPhase4.addPass(xla_jit::createTrimArgsPass());
}
