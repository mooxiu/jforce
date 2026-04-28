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
  std::unique_ptr<mlir::Pass> createTranslatePass(); // INFO: improved version for createWorkdistributeToStableHLOPass
  std::unique_ptr<mlir::Pass> createAffineCFGPass();
  std::unique_ptr<mlir::Pass> createCleanFIRLoopPass();

  void registerAnnotatePass();
  void registerAliasingPass();
  void registerShapeInferPass();
  void registerTrimArgsPass(); 
  void registerWorkdistributeToStableHLOPass();
  void registerTranslatePass();
  void registerAffineCFGPass();
  void registerCleanFIRLoopPass();

  void registerAllJitPasses();
}

#endif
