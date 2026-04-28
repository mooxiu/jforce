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

  void registerAnnotatePass();
  void registerAliasingPass();
  void registerShapeInferPass();
  void registerTrimArgsPass(); 
  void registerWorkdistributeToStableHLOPass();
  void registerAffineCFGPass();

  void registerAllJitPasses();
}

#endif
