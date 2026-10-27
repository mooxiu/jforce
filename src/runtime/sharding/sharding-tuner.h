#ifndef SHARDING_TUNER_H
#define SHARDING_TUNER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "runtime/sharding/sharding-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

// Returned layouts reference meshAxes; its storage must outlive them.
llvm::SmallVector<OperationShardingStrategy>
getMatmulStrategies(mlir::RankedTensorType lhsType,
                    mlir::RankedTensorType rhsType,
                    llvm::ArrayRef<MeshAxis> meshAxes);

class ShardingTuner {
public:
  void tune(MLIRContext &ctx,
            const llvm::DenseMap<size_t, func::FuncOp> &kernelsMap,
            const llvm::SmallVector<size_t> &series);
};

#endif
