#include "sharding-tuner.h"
#include "Highs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "runtime/sharding/sharding-codec.h"
#include "runtime/sharding/sharding-types.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
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
llvm::SmallVector<OpStrategy>
getMatmulStrategies(mlir::RankedTensorType lhsType,
                    mlir::RankedTensorType rhsType,
                    llvm::ArrayRef<MeshAxis> meshAxes) {
  if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() ||
      lhsType.getDimSize(1) != rhsType.getDimSize(0)) {
    return {};
  }

  // Start with one unsharded strategy, then assign every mesh axis.
  // Dimension sizes are assumed divisible by the axis sizes (padding later).
  OpStrategy initial;
  initial.operands = {TensorSharding(2), TensorSharding(2)}; // A and B
  initial.results = {TensorSharding(2)};                     // C
  llvm::SmallVector<OpStrategy> strategies{initial};

  for (const MeshAxis &axis : meshAxes) {
    if (axis.size <= 0) {
      return {};
    }
    if (axis.size == 1) {
      continue;
    }

    llvm::SmallVector<OpStrategy> next;
    for (const auto &strategy : strategies) {
      auto splitM = strategy;
      splitM.operands[0][0].push_back(axis); // A's M dimension
      splitM.results[0][0].push_back(axis);  // C's M dimension
      next.push_back(std::move(splitM));

      auto splitN = strategy;
      splitN.operands[1][1].push_back(axis); // B's N dimension
      splitN.results[0][1].push_back(axis);  // C's N dimension
      next.push_back(std::move(splitN));

      auto splitK = strategy;
      splitK.operands[0][1].push_back(axis); // A's K dimension
      splitK.operands[1][0].push_back(axis); // B's K dimension
      next.push_back(std::move(splitK));
    }
    strategies = std::move(next);
  }

  return strategies;
}

// Listing all shardings for a single tensor.
// Every sharding of one tensor: each mesh axis is either unused (replicated) or
// assigned to one dimension. Includes the fully replicated one; a scalar has
// only that one. Dimension sizes are assumed divisible by the axis sizes
// (padding later).
// Returned shardings reference meshAxes; its storage must outlive them.
static llvm::SmallVector<TensorSharding>
getShardings(mlir::RankedTensorType type, llvm::ArrayRef<MeshAxis> meshAxes) {
  // Start fully replicated.
  llvm::SmallVector<TensorSharding> shardings{TensorSharding(type.getRank())};

  for (const MeshAxis &axis : meshAxes) {
    llvm::SmallVector<TensorSharding> next;
    for (const auto &sharding : shardings) {
      // An unused axis remains replicated.
      next.push_back(sharding);

      for (int64_t dim = 0; dim < type.getRank(); ++dim) {
        auto sharded = sharding;
        sharded[dim].push_back(axis);
        next.push_back(std::move(sharded));
      }
    }
    shardings = std::move(next);
  }

  return shardings;
}

llvm::SmallVector<OpStrategy>
getEleBinaryOpStrategies(mlir::RankedTensorType lhsType,
                         mlir::RankedTensorType rhsType,
                         llvm::ArrayRef<MeshAxis> meshAxes) {
  // This version handles elementwise operations with equal, static shapes.
  if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || lhsType.getShape() != rhsType.getShape()) {
    return {};
  }

  // Both operands and the result use the same sharding.
  llvm::SmallVector<OpStrategy> strategies;
  for (const TensorSharding &sharding : getShardings(lhsType, meshAxes)) {
    OpStrategy strategy;
    strategy.operands = {sharding, sharding};
    strategy.results = {sharding};
    strategies.push_back(std::move(strategy));
  }
  return strategies;
}

bool keepEleBinaryOpStrategy(const OpStrategy &candidate,
                             const TensorSharding *lhsProducerSharding,
                             const TensorSharding *rhsProducerSharding) {
  if (!lhsProducerSharding && !rhsProducerSharding) {
    return true;
  }
  if (candidate.results.size() != 1) {
    return false;
  }
  // One producer: inherit its sharding. Two producers: their join.
  TensorSharding target =
      lhsProducerSharding ? *lhsProducerSharding : *rhsProducerSharding;
  if (lhsProducerSharding && rhsProducerSharding) {
    if (lhsProducerSharding->size() != rhsProducerSharding->size()) {
      return false;
    }
    target = join(*lhsProducerSharding, *rhsProducerSharding);
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
static OpStrategy getReplicatedStrategy(Operation *op) {
  auto shardingOf = [](Type type) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(type);
    return TensorSharding(tensorType ? tensorType.getRank() : 0);
  };
  OpStrategy strategy;
  for (Type type : op->getOperandTypes()) {
    strategy.operands.push_back(shardingOf(type));
  }
  for (Type type : op->getResultTypes()) {
    strategy.results.push_back(shardingOf(type));
  }
  return strategy;
}

static llvm::SmallVector<OpStrategy>
getOpStrategies(Operation *op, llvm::ArrayRef<MeshAxis> meshAxes) {
  llvm::SmallVector<OpStrategy> strategies;
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
                          const OpStrategy &strategy) {

  // TODO: this only considers the situation where one GPU can hold all the data.
  // For the situation where one GPU cannot hold all the data, consider Cannon's algorithm.
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

// True if `to` shards a dimension other than `dim` with `axis`.
static bool isOnOtherDim(const MeshAxis &axis, size_t dim,
                         const TensorSharding &to) {
  for (size_t d = 0; d < to.size(); ++d) {
    if (d == dim) {
      continue;
    }
    for (const auto &toAxis : to[d]) {
      if (toAxis.get().name == axis.name) {
        return true;
      }
    }
  }
  return false;
}

// Cost of converting one tensor of `type` from sharding `from` to sharding `to`.
// Each axis of `from` beyond joined = join(from, to) leaves its dimension:
// - moved: `to` uses it on another dimension, so an all-to-all over the moved
//   axes (m devices) sends (m-1)/m of the local shard.
// - gathered: `to` does not use it, so an all-gather over the gathered axes
//   (g devices) receives (g-1) local shards.
// The all-to-all goes first, while shards are small. Axes `to` adds are
// sliced locally, with no communication.
// e.g. a 12x12 f64 tensor on 3 devices, [{x}, {}] -> [{}, {x}]: x moves,
// m = 3, local shard 384 bytes, cost 2/3 * 384 = 256 bytes.
static double getReshardCost(const TensorSharding &from,
                                   const TensorSharding &to, Type type) {
  TensorSharding joined = join(from, to);
  int64_t moved = 1;    // devices in one all-to-all group
  int64_t gathered = 1; // devices in one all-gather group
  int64_t shards = 1;   // shards of the tensor at `from`
  for (size_t dim = 0; dim < from.size(); ++dim) {
    for (size_t i = 0; i < from[dim].size(); ++i) {
      const MeshAxis &axis = from[dim][i].get();
      shards *= axis.size;
      // joined[dim] is a prefix of from[dim]; those axes stay.
      if (i < joined[dim].size()) {
        continue;
      }
      if (isOnOtherDim(axis, dim, to)) {
        moved *= axis.size;
      } else {
        gathered *= axis.size;
      }
    }
  }
  double shardBytes = getBytes(type) / shards;
  return (moved - 1.0) / moved * shardBytes + (gathered - 1.0) * shardBytes;
}

// R_vu[i][j]: reshard cost of every value the edge carries.
static double getEdgeReshardCost(const Edge &edge,
                             const OpStrategy &src,
                             const OpStrategy &tgt) {
  double cost = 0;
  for (auto [resultIdx, operandIdx] : edge.valuePairs) {
    cost += getReshardCost(
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

// Add one node to the ILP: one binary column s_v,i per choice with cost
// costs[i], and the one-hot row sum_i s_v,i = 1. Returns the first column.
static HighsInt addIlpNode(Highs &highs, llvm::ArrayRef<double> costs) {
  HighsInt first = highs.getNumCol();
  for (double cost : costs) {
    HighsInt col = highs.getNumCol();
    highs.addCol(cost, 0, 1, 0, nullptr, nullptr);
    highs.changeColIntegrality(col, HighsVarType::kInteger);
  }
  std::vector<HighsInt> idx(costs.size());
  std::iota(idx.begin(), idx.end(), first);
  std::vector<double> ones(idx.size(), 1);
  highs.addRow(1, 1, idx.size(), idx.data(), ones.data());
  return first;
}

// Add the edge term sum_ij R_ij e_ij between node v (kv choices, columns from
// firstV) and node u (ku choices, columns from firstU). R is kv x ku, row-major.
// e_ij is continuous in [0, 1]; the constraints force e_ij = s_v,i * s_u,j.
static void addIlpEdge(Highs &highs, HighsInt firstV, size_t kv,
                       HighsInt firstU, size_t ku, llvm::ArrayRef<double> R) {
  // No resharding cost for any pair: the edge adds nothing.
  if (llvm::all_of(R, [](double c) { return c == 0; })) {
    return;
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
    rowIdx.push_back(firstV + i);
    rowVal.back() = -1;
    highs.addRow(-kHighsInf, 0, rowIdx.size(), rowIdx.data(), rowVal.data());
  }
  // Column j: sum_i e_ij - s_u,j <= 0
  for (size_t j = 0; j < ku; ++j) {
    std::vector<HighsInt> colIdx;
    std::vector<double> colVal(kv + 1, 1);
    for (size_t i = 0; i < kv; ++i) {
      colIdx.push_back(firstE + i * ku + j);
    }
    colIdx.push_back(firstU + j);
    colVal.back() = -1;
    highs.addRow(-kHighsInf, 0, colIdx.size(), colIdx.data(), colVal.data());
  }
}

// Run ILP model to find the best sharding strategy for each operation.
// min sum_v c_v^T s_v + sum_(v,u) sum_ij R_vu,ij e_vu,ij
// - v ranges over ops. Kernel arguments are replicated, so slicing them for any
//   user is local and costs nothing: they have no nodes or edges.
// - s_v binary and one-hot.
// - e_vu continuous in [0, 1]; the constraints force e_vu = s_v s_u^T.
ShardingPlan
Graph::tune(llvm::ArrayRef<MeshAxis> meshAxes) {
  Highs highs;
  highs.setOptionValue("output_flag", false);

  // Op nodes: one column per strategy, cost c_v.
  llvm::DenseMap<Operation *, llvm::SmallVector<OpStrategy>>
      strategies;
  llvm::DenseMap<Operation *, HighsInt> firstCol;
  for (Operation *op : opSet) {
    strategies[op] = getOpStrategies(op, meshAxes);
    llvm::SmallVector<double> costs;
    for (const auto &strategy : strategies[op]) {
      costs.push_back(getCommCost(op, strategy));
    }
    firstCol[op] = addIlpNode(highs, costs);
  }

  // Op -> op edges.
  for (const auto &entry : edgeMap) {
    for (const Edge &edge : entry.second) {
      const auto &src = strategies[edge.srcOp];
      const auto &tgt = strategies[edge.tgtOp];
      std::vector<double> R(src.size() * tgt.size());
      for (size_t i = 0; i < src.size(); ++i) {
        for (size_t j = 0; j < tgt.size(); ++j) {
          R[i * tgt.size() + j] = getEdgeReshardCost(edge, src[i], tgt[j]);
        }
      }
      addIlpEdge(highs, firstCol[edge.srcOp], src.size(),
                 firstCol[edge.tgtOp], tgt.size(), R);
    }
  }

  highs.run();
  if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
    llvm::errs() << "Sharding ILP is not solved to optimality!\n";
    std::exit(EXIT_FAILURE);
  }

  const auto &value = highs.getSolution().col_value;
  ShardingPlan chosen;
  for (Operation *op : opSet) {
    for (size_t i = 0; i < strategies[op].size(); ++i) {
      if (value[firstCol[op] + i] > 0.5) {
        chosen[op] = strategies[op][i];
      }
    }
  }
  return chosen;
}

Graph buildGraph(func::FuncOp kernel) {
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
  return graph;
}

double getPlanCost(const Graph &graph, const ShardingPlan &plan) {
  double cost = 0;
  for (Operation *op : graph.opSet) {
    cost += getCommCost(op, plan.at(op));
  }
  for (const auto &entry : graph.edgeMap) {
    for (const Edge &edge : entry.second) {
      cost += getEdgeReshardCost(edge, plan.at(edge.srcOp), plan.at(edge.tgtOp));
    }
  }
  return cost;
}

KernelIOShardings annotatePlan(func::FuncOp kernel, const ShardingPlan &plan,
                               llvm::StringRef meshName) {
  MLIRContext &ctx = *kernel.getContext();
  Block &entry = kernel.getBody().front();
  const unsigned numArgs = kernel.getNumArguments();
  auto ret = llvm::cast<func::ReturnOp>(entry.getTerminator());
  if (ret.getNumOperands() != numArgs) {
    kernel.emitError("expected one result per argument");
    std::exit(EXIT_FAILURE);
  }

  // Op results: the shardings chosen by the plan. XLA drops the sdy.sharding of
  // op results when it imports the module into Shardy; only returned values
  // keep theirs, as output shardings. So temporaries (not returned) get a
  // sharding constraint instead, added below because it inserts new ops.
  llvm::SmallVector<OpResult> temporaries;
  for (Operation &op : entry.without_terminator()) {
    auto it = plan.find(&op);
    if (it == plan.end()) {
      op.emitError("the sharding plan has no strategy for this operation");
      std::exit(EXIT_FAILURE);
    }
    const OpStrategy &strategy = it->second;
    for (unsigned j = 0; j < op.getNumResults(); j++) {
      OpResult result = op.getResult(j);
      // Scalars are always replicated, so they need no constraint.
      if (!strategy.results[j].empty() &&
        // TODO: currently the return operation use keep the last sharding by default,
        // So we do not add constraint before.
        // this might not be the best though. Check this decision later.
          !llvm::is_contained(result.getUsers(), ret.getOperation())) {
        temporaries.push_back(result);
        continue;
      }
      sdy::setSharding(result,
                       toShardingAttr(ctx, meshName, strategy.results[j]));
    }
  }

  OpBuilder builder(&ctx);
  for (OpResult result : temporaries) {
    builder.setInsertionPointAfter(result.getOwner());
    auto sharding = plan.at(result.getOwner()).results[result.getResultNumber()];
    auto constraint = sdy::ShardingConstraintOp::create(
        builder, result.getLoc(), result,
        toShardingAttr(ctx, meshName, sharding));
    result.replaceAllUsesExcept(constraint, constraint);
  }

  // Inputs are replicated: each user slices its operand locally, at no cost.
  // Outputs take the sharding of the value returned (the last write); an
  // argument returned as is stays replicated.
  KernelIOShardings ioShardings;
  for (unsigned i = 0; i < numArgs; i++) {
    auto type = llvm::cast<RankedTensorType>(kernel.getArgument(i).getType());
    TensorSharding output(type.getRank());
    if (auto result = llvm::dyn_cast<OpResult>(ret.getOperand(i))) {
      output = plan.at(result.getOwner()).results[result.getResultNumber()];
    }
    ioShardings.inputs.push_back(TensorSharding(type.getRank()));
    ioShardings.outputs.push_back(std::move(output));
  }

  // Annotate value shardings only; reshards to operand shardings are left to
  // Shardy/XLA.
  for (unsigned i = 0; i < numArgs; i++) {
    sdy::setSharding(kernel.getArgument(i),
                     toShardingAttr(ctx, meshName, ioShardings.inputs[i]));
    kernel.setResultAttr(
        i, sdy::TensorShardingAttr::name,
        toShardingAttr(ctx, meshName, ioShardings.outputs[i]));
    // Donation (set by AliasingPass) needs the input and output shards to have
    // the same size on each device, so drop it when their shardings differ.
    if (compare(ioShardings.inputs[i], ioShardings.outputs[i]) != 0) {
      kernel.removeArgAttr(i, "tf.aliasing_output");
    }
  }
  return ioShardings;
}

///
///
/// Sharding Tuner
///
///

llvm::DenseMap<size_t, ShardingPlan>
ShardingTuner::tune(MLIRContext &ctx,
                    const llvm::DenseMap<size_t, func::FuncOp> &kernelsMap,
                    const llvm::SmallVector<size_t> &series,
                    llvm::ArrayRef<MeshAxis> meshAxes) {
  llvm::DenseMap<size_t, ShardingPlan>
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
    chosenPerKernel[hash] = buildGraph(it->getSecond()).tune(meshAxes);
  }

  (void)ctx;
  return chosenPerKernel;
}
