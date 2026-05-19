#include "Passes.h"

void xla_jit::registerAllJitPasses() {
  registerAnnotatePass();
  registerAliasingPass();
  registerShapeInferPass();
  registerTrimArgsPass(); 
  registerWorkdistributeToStableHLOPass();   
  registerTranslatePass();
  registerAffineCFGPass();
  registerCleanFIRLoopPass();
  registerSimplifyDoLoopPass();
  registerAffineToStableHLORaisingPass();
  registerArithRaisingPass();
  registerOutlineAffinePass();
  registerPropagateConstantsPass();
  registerOptimizeMemOpsPass();
  registerCleanFIROpsPass();
}
