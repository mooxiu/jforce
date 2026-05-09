#ifndef PASSES_H
#define PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createAnnotatePass();
  std::unique_ptr<mlir::Pass> createAliasingPass(); 
  std::unique_ptr<mlir::Pass> createShapeInferPass(); 
  std::unique_ptr<mlir::Pass> createTrimArgsPass(); 
  std::unique_ptr<mlir::Pass> createWorkdistributeToStableHLOPass(); 
  std::unique_ptr<mlir::Pass> createAffineCFGPass();
  std::unique_ptr<mlir::Pass> createCleanFIRLoopPass();
  std::unique_ptr<mlir::Pass> createSimplifyDoLoopPass();
  std::unique_ptr<mlir::Pass> createAffineToStableHLORaisingPass();
  std::unique_ptr<mlir::Pass> createFIRLoadToAffineLoadPass();
  std::unique_ptr<mlir::Pass> createFirLICMPass();  
  std::unique_ptr<mlir::Pass> createArithRaisingPass();
  std::unique_ptr<mlir::Pass> createOutlineAffinePass();
  std::unique_ptr<mlir::Pass> createPropagateConstantsPass();

  void registerAnnotatePass();
  void registerAliasingPass();
  void registerShapeInferPass();
  void registerTrimArgsPass(); 
  void registerWorkdistributeToStableHLOPass();
  void registerAffineCFGPass();
  void registerCleanFIRLoopPass();
  void registerSimplifyDoLoopPass();
  void registerAffineToStableHLORaisingPass();
  void registerFIRLoadToAffineLoadPass();
  void registerFirLICMPass();
  void registerArithRaisingPass();
  void registerOutlineAffinePass();
  void registerPropagateConstantsPass();

  void registerAllJitPasses();
}

#endif
