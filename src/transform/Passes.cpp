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
  registerFIRLoadToAffineLoadPass();
  registerFirLICMPass();
  registerArithRaisingPass();
  registerOutlineAffinePass();
  registerPropagateConstantsPass();
  registerCleanFIRLoadPass();
  registerCleanFIROpsPass();
}
