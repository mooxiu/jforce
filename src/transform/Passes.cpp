#include "Passes.h"

void xla_jit::registerAllJitPasses() {
  registerCleanTempsPass();
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
  registerLoopSinkingPass();
  registerRemergePass();
  registerIfConversionPass();
  registerPolygeistMem2RegPass();
  registerFoldSCFIfPass();
  registerFoldRepeatConversionsPass();
}
