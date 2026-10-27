#include "sharding-tuner.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "runtime/sharding/sharding-types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <utility>

llvm::SmallVector<ArgSharding>
getCandidates(mlir::RankedTensorType type, llvm::ArrayRef<MeshAxis> meshAxes) {
  auto rank = type.getRank();
  // Suppose we only shard on one dimension at most with one axis
  llvm::SmallVector<ArgSharding> candidates;
  // case 1: no sharding
  ArgSharding argSharding;
  argSharding.resize(rank);
  candidates.push_back(std::move(argSharding));

  // case 2: sharding on dimIdx of the argument by axisIdx of the deviceMesh
  for (int axisIdx = 0; axisIdx < meshAxes.size(); axisIdx++) {
    for (int dimIdx = 0; dimIdx < rank; dimIdx++) {
      ArgSharding argSharding;
      argSharding.resize(rank);
      argSharding[dimIdx].push_back(meshAxes[axisIdx]);
      candidates.push_back(std::move(argSharding));
    }
  }

  return candidates;
}

llvm::SmallVector<ShardingDecision>
getKernelCandidates(func::FuncOp kernel, llvm::ArrayRef<MeshAxis> meshAxes) {
  // Init with default plan: all replica
  llvm::SmallVector<ShardingDecision> plans(1);

  // adding non-all-replica plans
  for (auto type : kernel.getArgumentTypes()) {
    auto argCandidates =
        getCandidates(llvm::cast<RankedTensorType>(type), meshAxes);
    llvm::SmallVector<ShardingDecision> next;

    for (const auto &plan : plans) {
      for (const auto &candidate : argCandidates) {
        auto extended = plan;
        extended.push_back(candidate);
        next.push_back(std::move(extended));
      }
    }
    plans = std::move(next);
  }
  return plans;
}

// Generate strategies for an ordinary 2D matmul A[M,K] x B[K,N].
llvm::SmallVector<OperationShardingStrategy> getMatmulStrategies(
    mlir::RankedTensorType lhsType, mlir::RankedTensorType rhsType,
    llvm::ArrayRef<MeshAxis> meshAxes) {
  if (!lhsType || !rhsType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return {};

  const int64_t M = lhsType.getDimSize(0);
  const int64_t K = lhsType.getDimSize(1);
  const int64_t N = rhsType.getDimSize(1);

  // Start with one unsharded strategy, then assign every mesh axis.
  OperationShardingStrategy initial;
  initial.operands = {ArgSharding(2), ArgSharding(2)}; // A and B
  initial.results = {ArgSharding(2)};                 // C
  llvm::SmallVector<OperationShardingStrategy> strategies{initial};

  // Previous axes have already divided the dimension evenly.
  auto canSplit = [](int64_t size, const DimSharding &previousAxes,
                     int64_t axisSize) {
    for (const auto &previousAxis : previousAxes)
      size /= previousAxis.get().size;
    return size % axisSize == 0;
  };

  for (const MeshAxis &axis : meshAxes) {
    if (axis.size <= 0)
      return {};
    if (axis.size == 1)
      continue;

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


void ShardingTuner::tune(MLIRContext &ctx,
                         const llvm::DenseMap<size_t, func::FuncOp> &kernelsMap,
                         const llvm::SmallVector<size_t> &series) {
  llvm::SmallVector<func::FuncOp> executionKernelTrace;
  executionKernelTrace.reserve(series.size());
  for (const size_t &hash : series) {
    auto it = kernelsMap.find(hash);
    if (it == kernelsMap.end()) {
      llvm::errs() << "Hash not found in kernels Map!\n";
      std::exit(EXIT_FAILURE);
    }
    executionKernelTrace.push_back(it->getSecond());
  }

  (void)ctx; // Tuning is not implemented yet.
}
