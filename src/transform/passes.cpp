#include "passes.h"

void xla_jit::registerAllJitPasses() {
  registerAliasingPass();
  registerShapeInferPass();
  registerTrimArgsPass(); 
  registerWorkdistributeToStableHLOPass();   
}
