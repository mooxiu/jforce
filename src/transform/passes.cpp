#include "passes.h"

void xla_jit::registerAllJitPasses() {
  registerAnnotatePass();
  registerAliasingPass();
  registerShapeInferPass();
  registerTrimArgsPass(); 
  registerWorkdistributeToStableHLOPass();   
  registerTranslatePass();
  registerAffineCFGPass();
}
