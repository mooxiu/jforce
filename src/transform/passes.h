#ifndef PASSES_H
#define PASSES_H

#include "mlir/Pass/Pass.h"

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createAliasingPass(); 
  std::unique_ptr<mlir::Pass> createShapeInferPass(); 
  std::unique_ptr<mlir::Pass> createTrimArgsPass(); 
  std::unique_ptr<mlir::Pass> createWorkdistributeToStableHLOPass(); 

  void registerAliasingPass();
  void registerShapeInferPass();
  void registerTrimArgsPass(); 
  void registerWorkdistributeToStableHLOPass();

  void registerAllJitPasses();
}

#endif
