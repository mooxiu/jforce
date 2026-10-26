#ifndef SHARDING_TUNER_H
#define SHARDING_TUNER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

class ShardingTuner {
public:
  void tune(MLIRContext &ctx,
            const llvm::DenseMap<size_t, func::FuncOp> &kernelsMap,
            const llvm::SmallVector<size_t> &series);
};

#endif
