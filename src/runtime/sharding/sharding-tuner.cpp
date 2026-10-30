#include "sharding-tuner.h"
#include "Highs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "runtime/sharding/sharding-types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>

// Generate strategies for an ordinary 2D matmul A[M,K] x B[K,N].
// Consider the device mesh to be m-dimensional, then the sharding space for
// this operator is 3^m.
// - 3: there are 3 unique dimension in Matmul inputs: M, K, N; and as we know
// it can be written as a 3 layer nested loop
// - m: for each M, K, N, it can be sharded by one of the device dimension, and
// we have to apply all of the device dimension (else there will be repeated
// computation).
llvm::SmallVector<OperationShardingStrategy>
getMatmulStrategies(mlir::RankedTensorType lhsType,
                    mlir::RankedTensorType rhsType,
                    llvm::ArrayRef<MeshAxis> meshAxes) {
  if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() ||
      lhsType.getDimSize(1) != rhsType.getDimSize(0)) {
    return {};
  }

  const int64_t M = lhsType.getDimSize(0);
  const int64_t K = lhsType.getDimSize(1);
  const int64_t N = rhsType.getDimSize(1);

  // Start with one unsharded strategy, then assign every mesh axis.
  OperationShardingStrategy initial;
  initial.operands = {ArgSharding(2), ArgSharding(2)}; // A and B
  initial.results = {ArgSharding(2)};                  // C
  llvm::SmallVector<OperationShardingStrategy> strategies{initial};

  // Previous axes have already divided the dimension evenly.
  auto canSplit = [](int64_t size, const DimSharding &previousAxes,
                     int64_t axisSize) {
    for (const auto &previousAxis : previousAxes) {
      size /= previousAxis.get().size;
    }
    return size % axisSize == 0;
  };

  for (const MeshAxis &axis : meshAxes) {
    if (axis.size <= 0) {
      return {};
    }
    if (axis.size == 1) {
      continue;
    }

    llvm::SmallVector<OperationShardingStrategy> next;
    for (const auto &strategy : strategies) {
      if (canSplit(M, strategy.operands[0][0], axis.size)) {
        auto splitM = strategy;
        splitM.operands[0][0].push_back(axis); // A's M dimension
        splitM.results[0][0].push_back(axis);  // C's M dimension
        next.push_back(std::move(splitM));
      }
      if (canSplit(N, strategy.operands[1][1], axis.size)) {
        auto splitN = strategy;
        splitN.operands[1][1].push_back(axis); // B's N dimension
        splitN.results[0][1].push_back(axis);  // C's N dimension
        next.push_back(std::move(splitN));
      }
      if (canSplit(K, strategy.operands[0][1], axis.size)) {
        auto splitK = strategy;
        splitK.operands[0][1].push_back(axis); // A's K dimension
        splitK.operands[1][0].push_back(axis); // B's K dimension
        next.push_back(std::move(splitK));
      }
    }
    strategies = std::move(next);
  }

  return strategies;
}

// Element-wise, binary operation.
// For example: add, multiply, etc.
// They are lightweight, operations.
// Replica is allowed.
llvm::SmallVector<OperationShardingStrategy>
getEleBinaryOpStrategies(mlir::RankedTensorType lhsType,
                         mlir::RankedTensorType rhsType,
                         llvm::ArrayRef<MeshAxis> meshAxes) {
  // This version handles elementwise operations with equal, static shapes.
  if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || lhsType.getShape() != rhsType.getShape()) {
    return {};
  }

  // Both operands and the result start fully replicated.
  ArgSharding replicated(lhsType.getRank());
  OperationShardingStrategy initial;
  initial.operands = {replicated, replicated};
  initial.results = {replicated};
  llvm::SmallVector<OperationShardingStrategy> strategies{initial};

  for (const MeshAxis &axis : meshAxes) {
    if (axis.size <= 0) {
      return {};
    }
    if (axis.size == 1) {
      continue;
    }

    llvm::SmallVector<OperationShardingStrategy> next;
    for (const auto &strategy : strategies) {
      // An unused axis remains replicated.
      next.push_back(strategy);

      for (int64_t dim = 0; dim < lhsType.getRank(); ++dim) {
        int64_t remainingSize = lhsType.getDimSize(dim);
        for (const auto &previousAxis : strategy.results[0][dim]) {
          remainingSize /= previousAxis.get().size;
        }
        if (remainingSize % axis.size != 0) {
          continue;
        }

        auto sharded = strategy;
        sharded.operands[0][dim].push_back(axis);
        sharded.operands[1][dim].push_back(axis);
        sharded.results[0][dim].push_back(axis);
        next.push_back(std::move(sharded));
      }
    }
    strategies = std::move(next);
  }

  return strategies;
}

bool keepEleBinaryOpStrategy(const OperationShardingStrategy &candidate,
                             const ArgSharding *lhsProducerLayout,
                             const ArgSharding *rhsProducerLayout) {
  if (!lhsProducerLayout && !rhsProducerLayout) {
    return true;
  }
  if (candidate.results.size() != 1) {
    return false;
  }
  // One producer: inherit its layout. Two producers: their join.
  ArgSharding target =
      lhsProducerLayout ? *lhsProducerLayout : *rhsProducerLayout;
  if (lhsProducerLayout && rhsProducerLayout) {
    if (lhsProducerLayout->size() != rhsProducerLayout->size()) {
      return false;
    }
    target = join(*lhsProducerLayout, *rhsProducerLayout);
  }
  return compare(candidate.results[0], target) == 0;
}

///
///
/// Graph
///
///

static double getBytes(Type type) {
  auto tensorType = llvm::cast<RankedTensorType>(type);
  return static_cast<double>(tensorType.getNumElements()) *
         tensorType.getElementTypeBitWidth() / 8.0;
}

static bool isMatmul(Operation *op) {
  auto dot = llvm::dyn_cast<stablehlo::DotGeneralOp>(op);
  if (!dot) {
    return false;
  }
  auto dims = dot.getDotDimensionNumbers();
  return dims.getLhsBatchingDimensions().empty() &&
         dims.getRhsBatchingDimensions().empty() &&
         dims.getLhsContractingDimensions() == llvm::ArrayRef<int64_t>{1} &&
         dims.getRhsContractingDimensions() == llvm::ArrayRef<int64_t>{0};
}

// Fallback for unsupported ops: every operand and result replicated.
static OperationShardingStrategy getReplicatedStrategy(Operation *op) {
  auto layoutOf = [](Type type) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(type);
    return ArgSharding(tensorType ? tensorType.getRank() : 0);
  };
  OperationShardingStrategy strategy;
  for (Type type : op->getOperandTypes()) {
    strategy.operands.push_back(layoutOf(type));
  }
  for (Type type : op->getResultTypes()) {
    strategy.results.push_back(layoutOf(type));
  }
  return strategy;
}

static llvm::SmallVector<OperationShardingStrategy>
getOpStrategies(Operation *op, llvm::ArrayRef<MeshAxis> meshAxes) {
  llvm::SmallVector<OperationShardingStrategy> strategies;
  if (isMatmul(op)) {
    strategies = getMatmulStrategies(
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType()),
        llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType()),
        meshAxes);
  } else if (op->hasTrait<OpTrait::Elementwise>() &&
             op->getNumOperands() == 2) {
    strategies = getEleBinaryOpStrategies(
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType()),
        llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType()),
        meshAxes);
  }
  // Unsupported ops, or shapes the generators reject, stay replicated.
  if (strategies.empty()) {
    strategies.push_back(getReplicatedStrategy(op));
  }
  return strategies;
}

// c_v: a sharded K dimension leaves partial sums of C, which need an
// all-reduce over those axes: 2(n-1)/n * bytes of the local C shard.
static double getCommCost(Operation *op,
                          const OperationShardingStrategy &strategy) {

  auto getDotGeneralCost = [&](Operation *op) -> double {
    if (!isMatmul(op) || strategy.operands.size() != 2) {
      return 0;
    }
    int64_t n = 1;
    for (const auto &axis : strategy.operands[0][1]) {
      n *= axis.get().size;
    }
    if (n == 1) {
      return 0;
    }
    // shardBytes is the bytes on each device.
    // Consider a 2x2 device mesh, and a matrix multiplication C = A x B.
    //
    // Sharding:
    // - sharding A's dim-1 with axis-0. Each partition is M x (K/2)
    // - sharding B's dim-0 with axis-1. Each partition is (K/2) x N
    // Therefore, on each partition, it will get a M x N matrix.
    //
    // Cost:
    // Reduce-Scatter: chunk each part in device size: N, pass around the each
    // part (N - 1), therefore sending (N-1)/N All-gather: send added part to
    // all, again, (N-1)/N In total: 2(N-1)/N
    double shardBytes = getBytes(op->getResult(0).getType());
    for (const auto &dim : strategy.results[0]) {
      for (const auto &axis : dim) {
        shardBytes /= axis.get().size;
      }
    }
    return 2.0 * (n - 1) / n * shardBytes;
  };

  auto cost = llvm::TypeSwitch<Operation *, double>(op)
      .Case<stablehlo::DotGeneralOp>(
          [&](auto) { return getDotGeneralCost(op); })
      .Case<stablehlo::AddOp, stablehlo::SubtractOp, stablehlo::MulOp>(
          // If we have inputs of different inputs, we should first reshard them
          // to be compatible. Therefore the cost is all in resharding cost.
          [&](auto) { return 0; })
      .Default([&](auto) {
        llvm::errs() << "Unexpected stablehlo operatior: ";
        op->print(llvm::errs());
        std::exit(EXIT_FAILURE);
        return 0;        
      });
  return cost;
}

// Cost of converting one tensor of `type` from layout `from` to layout `to`.
// 1. Go up to joined = join(from, to): all-gather the axes `from` has beyond
//    `joined`. This is the only step with communication.
// 2. Go down from `joined` to `to`: every device already holds the data, so
//    it slices locally.
// A ring all-gather over n devices sends (n-1)/n of the joined shard.
static double getLayoutReshardCost(const ArgSharding &from,
                                   const ArgSharding &to, Type type) {
  ArgSharding joined = join(from, to);
  int64_t groupSize = 1;    // devices in one all-gather group
  int64_t joinedShards = 1; // shards of the tensor at the joined layout
  for (size_t dim = 0; dim < from.size(); ++dim) {
    // joined[dim] is a prefix of from[dim]; the remaining axes are gathered.
    for (size_t i = 0; i < from[dim].size(); ++i) {
      int64_t axisSize = from[dim][i].get().size;
      if (i < joined[dim].size()) {
        joinedShards *= axisSize;
      } else {
        groupSize *= axisSize;
      }
    }
  }
  // `to` is the same as or finer than `from`: local slicing only.
  if (groupSize == 1) {
    return 0;
  }
  double joinedShardBytes = getBytes(type) / joinedShards;
  return (groupSize - 1.0) / groupSize * joinedShardBytes;
}

// R_vu[i][j]: reshard cost of every value the edge carries.
static double getReshardCost(const Edge &edge,
                             const OperationShardingStrategy &src,
                             const OperationShardingStrategy &tgt) {
  double cost = 0;
  for (auto [resultIdx, operandIdx] : edge.valuePairs) {
    cost += getLayoutReshardCost(
        src.results[resultIdx], tgt.operands[operandIdx],
        edge.srcOp->getResult(resultIdx).getType());
  }
  return cost;
}

void Graph::addOp(Operation *op) { opSet.insert(op); }

void Graph::addEdge(Operation *srcOp, Operation *tgtOp,
                    std::pair<int, int> valuePair) {
  auto &edges = edgeMap[srcOp];
  for (Edge &edge : edges) {
    if (edge.tgtOp == tgtOp) {
      edge.valuePairs.push_back(valuePair);
      return;
    }
  }
  edges.push_back(Edge{srcOp, tgtOp, {valuePair}});
}

// min sum_v c_v^T s_v + sum_(v,u) sum_ij R_vu,ij e_vu,ij
// - s_v binary and one-hot.
// - e_vu continuous in [0, 1]; the constraints below force e_vu = s_v s_u^T.
llvm::DenseMap<Operation *, OperationShardingStrategy>
Graph::tune(llvm::ArrayRef<MeshAxis> meshAxes) {
  Highs highs;
  highs.setOptionValue("output_flag", false);

  // s_v: one column per strategy, first column index kept per op.
  llvm::DenseMap<Operation *, llvm::SmallVector<OperationShardingStrategy>>
      strategies;
  llvm::DenseMap<Operation *, HighsInt> firstCol;
  for (Operation *op : opSet) {
    strategies[op] = getOpStrategies(op, meshAxes);
    const auto &opStrategies = strategies[op];
    HighsInt first = highs.getNumCol();
    firstCol[op] = first;
    for (const auto &strategy : opStrategies) {
      HighsInt col = highs.getNumCol();
      highs.addCol(getCommCost(op, strategy), 0, 1, 0, nullptr, nullptr);
      highs.changeColIntegrality(col, HighsVarType::kInteger);
    }
    // sum_i s_v,i = 1
    std::vector<HighsInt> idx(opStrategies.size());
    std::iota(idx.begin(), idx.end(), first);
    std::vector<double> ones(idx.size(), 1);
    highs.addRow(1, 1, idx.size(), idx.data(), ones.data());
  }

  for (const auto &entry : edgeMap) {
    for (const Edge &edge : entry.second) {
      const auto &src = strategies[edge.srcOp];
      const auto &tgt = strategies[edge.tgtOp];
      const size_t kv = src.size(), ku = tgt.size();

      std::vector<double> R(kv * ku);
      for (size_t i = 0; i < kv; ++i) {
        for (size_t j = 0; j < ku; ++j) {
          R[i * ku + j] = getReshardCost(edge, src[i], tgt[j]);
        }
      }
      // No resharding cost for any pair: the edge adds nothing.
      if (llvm::all_of(R, [](double c) { return c == 0; })) {
        continue;
      }

      HighsInt firstE = highs.getNumCol();
      for (double cost : R) {
        highs.addCol(cost, 0, 1, 0, nullptr, nullptr);
      }

      // sum_ij e_ij = 1
      std::vector<HighsInt> idx(R.size());
      std::iota(idx.begin(), idx.end(), firstE);
      std::vector<double> ones(idx.size(), 1);
      highs.addRow(1, 1, idx.size(), idx.data(), ones.data());

      // Row i: sum_j e_ij - s_v,i <= 0
      for (size_t i = 0; i < kv; ++i) {
        std::vector<HighsInt> rowIdx;
        std::vector<double> rowVal(ku + 1, 1);
        for (size_t j = 0; j < ku; ++j) {
          rowIdx.push_back(firstE + i * ku + j);
        }
        rowIdx.push_back(firstCol[edge.srcOp] + i);
        rowVal.back() = -1;
        highs.addRow(-kHighsInf, 0, rowIdx.size(), rowIdx.data(),
                     rowVal.data());
      }
      // Column j: sum_i e_ij - s_u,j <= 0
      for (size_t j = 0; j < ku; ++j) {
        std::vector<HighsInt> colIdx;
        std::vector<double> colVal(kv + 1, 1);
        for (size_t i = 0; i < kv; ++i) {
          colIdx.push_back(firstE + i * ku + j);
        }
        colIdx.push_back(firstCol[edge.tgtOp] + j);
        colVal.back() = -1;
        highs.addRow(-kHighsInf, 0, colIdx.size(), colIdx.data(),
                     colVal.data());
      }
    }
  }

  highs.run();
  if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
    llvm::errs() << "Sharding ILP is not solved to optimality!\n";
    std::exit(EXIT_FAILURE);
  }

  const auto &value = highs.getSolution().col_value;
  llvm::DenseMap<Operation *, OperationShardingStrategy> chosen;
  for (Operation *op : opSet) {
    for (size_t i = 0; i < strategies[op].size(); ++i) {
      if (value[firstCol[op] + i] > 0.5) {
        chosen[op] = strategies[op][i];
      }
    }
  }
  return chosen;
}

///
///
/// Sharding Tuner
///
///

llvm::DenseMap<size_t, llvm::DenseMap<Operation *, OperationShardingStrategy>>
ShardingTuner::tune(MLIRContext &ctx,
                    const llvm::DenseMap<size_t, func::FuncOp> &kernelsMap,
                    const llvm::SmallVector<size_t> &series,
                    llvm::ArrayRef<MeshAxis> meshAxes) {
  llvm::DenseMap<size_t, llvm::DenseMap<Operation *, OperationShardingStrategy>>
      chosenPerKernel;
  for (const size_t &hash : series) {
    // A kernel repeated in the series is the same function: tune it once.
    if (chosenPerKernel.contains(hash)) {
      continue;
    }
    auto it = kernelsMap.find(hash);
    if (it == kernelsMap.end()) {
      llvm::errs() << "Hash not found in kernels Map!\n";
      std::exit(EXIT_FAILURE);
    }
    func::FuncOp kernel = it->getSecond();

    // Ops in a block are already in topological order. Collect ops and edges.
    Graph graph;
    for (Operation &op : kernel.getBody().getOps()) {
      if (llvm::isa<func::ReturnOp>(op)) {
        break;
      }
      graph.addOp(&op);
      for (OpOperand &operand : op.getOpOperands()) {
        auto result = llvm::dyn_cast<OpResult>(operand.get());
        if (!result) { // block argument, no producer op
          continue;
        }
        graph.addEdge(result.getOwner(), &op,
                      {static_cast<int>(result.getResultNumber()),
                       static_cast<int>(operand.getOperandNumber())});
      }
    }
    chosenPerKernel[hash] = graph.tune(meshAxes);
  }

  (void)ctx;
  return chosenPerKernel;
}
