#ifndef PASSES_H
#define PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createMemOpsFoldingPass();
  std::unique_ptr<mlir::Pass> createCleanTempsPass();
  std::unique_ptr<mlir::Pass> createAnnotatePass();
  std::unique_ptr<mlir::Pass> createAliasingPass(); 
  std::unique_ptr<mlir::Pass> createShapeInferPass(); 
  std::unique_ptr<mlir::Pass> createTrimArgsPass(); 
  std::unique_ptr<mlir::Pass> createWorkdistributeToStableHLOPass(); 
  std::unique_ptr<mlir::Pass> createAffineCFGPass();
  std::unique_ptr<mlir::Pass> createCleanFIRLoopPass();
  [[deprecated("Use CleanFIRLoopPass and other passes")]]
  std::unique_ptr<mlir::Pass> createSimplifyDoLoopPass();
  std::unique_ptr<mlir::Pass> createAffineToStableHLORaisingPass();
  std::unique_ptr<mlir::Pass> createFirLICMPass();  
  std::unique_ptr<mlir::Pass> createArithRaisingPass();
  std::unique_ptr<mlir::Pass> createOutlineAffinePass();
  std::unique_ptr<mlir::Pass> createPropagateConstantsPass();
  std::unique_ptr<mlir::Pass> createCleanFIRLoadPass();
  std::unique_ptr<mlir::Pass> createCleanFIROpsPass();
  [[deprecated("Should use MemOpsFoldingPass instead")]]
  std::unique_ptr<mlir::Pass> createOptimizeMemOpsPass();
  std::unique_ptr<mlir::Pass> createLoopSinkingPass();
  std::unique_ptr<mlir::Pass> createRemergePass();
  [[deprecated("Should use FoldSCFIfPass of EnzymeJAX")]]
  std::unique_ptr<mlir::Pass> createIfConversionPass();
  std::unique_ptr<mlir::Pass> createPolygeistMem2RegPass();
  std::unique_ptr<mlir::Pass> createFoldSCFIfPass();
  std::unique_ptr<mlir::Pass> createFoldRepeatConversionsPass();


  void registerMemOpsFoldingPass();
  void registerCleanTempsPass();
  void registerAnnotatePass();
  void registerAliasingPass();
  void registerShapeInferPass();
  void registerTrimArgsPass(); 
  void registerWorkdistributeToStableHLOPass();
  void registerAffineCFGPass();
  void registerCleanFIRLoopPass();
  [[deprecated("Use CleanFIRLoopPass and other passes")]]
  void registerSimplifyDoLoopPass();
  void registerAffineToStableHLORaisingPass();
  void registerFirLICMPass();
  void registerArithRaisingPass();
  void registerOutlineAffinePass();
  void registerPropagateConstantsPass();
  void registerCleanFIROpsPass();
  [[deprecated("Should use MemOpsFoldingPass instead")]]
  void registerOptimizeMemOpsPass();
  void registerAllJitPasses();
  void registerLoopSinkingPass();
  void registerRemergePass();
  [[deprecated("Should use FoldSCFIfPass of EnzymeJAX")]]
  void registerIfConversionPass();
  void registerPolygeistMem2RegPass();
  void registerFoldSCFIfPass();
  void registerFoldRepeatConversionsPass();
}

#endif
