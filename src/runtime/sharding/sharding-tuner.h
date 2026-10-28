#ifndef SHARDING_TUNER_H
#define SHARDING_TUNER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "runtime/sharding/sharding-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

using namespace mlir;

// Returned layouts reference meshAxes; its storage must outlive them.
llvm::SmallVector<OperationShardingStrategy>
getMatmulStrategies(mlir::RankedTensorType lhsType,
                    mlir::RankedTensorType rhsType,
                    llvm::ArrayRef<MeshAxis> meshAxes);

// Same-shaped elementwise operands and result use the same layout.
// Returned layouts reference meshAxes; its storage must outlive them.
llvm::SmallVector<OperationShardingStrategy>
getEleBinaryOpStrategies(mlir::RankedTensorType lhsType,
                         mlir::RankedTensorType rhsType,
                         llvm::ArrayRef<MeshAxis> meshAxes);

// Optional heuristic: nullptr means the operand has no selected producer.
// Inherit one producer, or keep equal dimensions and replicate conflicts.
bool keepEleBinaryOpStrategy(const OperationShardingStrategy &candidate,
                              const ArgSharding *lhsProducerLayout,
                              const ArgSharding *rhsProducerLayout);

/// Edge is defined between operations.
/// - srcOp: source operation.
/// - tgtOp: target opertaion.
/// - valuePairs: a pair of source operation's result idx to target operand's idx.
struct Edge {
  Operation* srcOp;
  Operation* tgtOp;
  llvm::SmallVector<std::pair<int, int>> valuePairs;
};

struct Graph {
  llvm::DenseSet<Operation*> opSet;
  // Keyed by srcOp: all outgoing edges of that operation.
  llvm::DenseMap<Operation*, llvm::SmallVector<Edge>> edgeMap;

  // Solve the ILP over this graph and return the chosen strategy of each op.
  // Returned layouts reference meshAxes; its storage must outlive them.
  llvm::DenseMap<Operation*, OperationShardingStrategy>
  tune(llvm::ArrayRef<MeshAxis> meshAxes);
};

class ShardingTuner {
private:
  Graph graph;
  void addEdge(Operation* srcOp, Operation* tgtOp, std::pair<int, int> valuePair);
  void addOp(Operation* op);

public:
  void tune(MLIRContext &ctx,
            const llvm::DenseMap<size_t, func::FuncOp> &kernelsMap,
            const llvm::SmallVector<size_t> &series);
};

#endif
