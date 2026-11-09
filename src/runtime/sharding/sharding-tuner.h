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

// The solution of the ILP for one kernel: the strategy chosen for each op.
using ShardingPlan = llvm::DenseMap<Operation *, OpStrategy>;

// Returned shardings reference meshAxes; its storage must outlive them.
llvm::SmallVector<OpStrategy>
getMatmulStrategies(mlir::RankedTensorType lhsType,
                    mlir::RankedTensorType rhsType,
                    llvm::ArrayRef<MeshAxis> meshAxes);

// Same-shaped elementwise operands and result use the same sharding.
// Returned shardings reference meshAxes; its storage must outlive them.
llvm::SmallVector<OpStrategy>
getEleBinaryOpStrategies(mlir::RankedTensorType lhsType,
                         mlir::RankedTensorType rhsType,
                         llvm::ArrayRef<MeshAxis> meshAxes);

// TODO: this is only used in tests for now.
// Optional heuristic: nullptr means the operand has no selected producer.
// Keep the candidate whose result is the producer's sharding, or the join of
// both producers' shardings (see the sharding lattice in sharding-types.h).
bool keepEleBinaryOpStrategy(const OpStrategy &candidate,
                             const TensorSharding *lhsProducerSharding,
                             const TensorSharding *rhsProducerSharding);

/// Edge is defined between operations.
/// - srcOp: source operation.
/// - tgtOp: target opertaion.
/// - valuePairs: a pair of source operation's result idx to target operand's
/// idx.
///
/// Example:
///   %c = dot_general %a, %b // srcOp
///   %d = dot_general %c, %c // tgtOp
/// In this example, we are supposed to have 2 pairs {0, 0} and {0, 1}
struct Edge {
  Operation *srcOp;
  Operation *tgtOp;
  llvm::SmallVector<std::pair<int, int>> valuePairs;
};

struct Graph {
  llvm::DenseSet<Operation *> opSet;

  // Keyed by srcOp: all outgoing edges of that operation.
  llvm::DenseMap<Operation *, llvm::SmallVector<Edge>> edgeMap;

  void addOp(Operation *op);
  void addEdge(Operation *srcOp, Operation *tgtOp,
               std::pair<int, int> valuePair);

  // Solve the ILP over this graph and return the chosen strategy of each op.
  // Returned shardings reference meshAxes; its storage must outlive them.
  ShardingPlan tune(llvm::ArrayRef<MeshAxis> meshAxes);
};

// One graph per kernel: every op before func.return and the edges between them.
// Kernel arguments are replicated, so they need no nodes.
Graph buildGraph(func::FuncOp kernel);

// Objective value of a plan: sum of c_v over ops plus R_vu over edges.
// The plan must contain every op of the graph.
double getPlanCost(const Graph &graph, const ShardingPlan &plan);

// Annotate the value sharding of every value: op results (sdy.sharding, or a
// sdy.sharding_constraint for temporaries, i.e. results not returned), kernel
// arguments (inputs, all replicated), and kernel results (outputs, the
// sharding of the value returned). It is done in a declarative way: operand
// shardings are not annotated, and the reshards they need are left to
// Shardy/XLA.
KernelIOShardings annotatePlan(func::FuncOp kernel, const ShardingPlan &plan,
                               llvm::StringRef meshName);

class ShardingTuner {
public:
  // Tune each kernel in series once, one graph per kernel.
  // Returns the chosen strategy of every op, per kernel hash.
  // Returned shardings reference meshAxes; its storage must outlive them.
  llvm::DenseMap<size_t, ShardingPlan>
  tune(MLIRContext &ctx, const llvm::DenseMap<size_t, func::FuncOp> &kernelsMap,
       const llvm::SmallVector<size_t> &series,
       llvm::ArrayRef<MeshAxis> meshAxes);
};

#endif
