#include "Passes.h"

void xla_jit::registerAllJitPasses() {
  registerAnnotatePass();
  registerAliasingPass();
  registerShapeInferPass();
  registerTrimArgsPass(); 
  registerWorkdistributeToStableHLOPass();   
  registerAffineCFGPass();
  registerCleanFIRLoopPass();
  registerSimplifyDoLoopPass();
  registerAffineToStableHLORaisingPass();
  registerArithRaisingPass();
  registerOutlineAffinePass();
  registerPropagateConstantsPass();
  registerOptimizeMemOpsPass();
  registerCleanFIROpsPass();
  registerLoopSinkingPass();
  registerRemergePass();
}
