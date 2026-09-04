#ifndef PIPELINS_H 
#define PIPELINS_H

#include "mlir/Pass/PassManager.h"

using namespace mlir;

void createLowerToStableHLOPassPipeline(PassManager& pm);

#endif
