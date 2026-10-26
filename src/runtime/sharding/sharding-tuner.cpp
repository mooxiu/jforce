#include "sharding-tuner.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "runtime/sharding/sharding-types.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <shardy/dialect/sdy/transforms/propagation/passes.h>
#include <utility>

///
/// module {
///   sdy.mesh @dummy = <["x"=2, "y"=2]>
///   func.func @main(%arg0: tensor<8x8xf64> {sdy.sharding = #sdy.sharding<@dummy, [{}, {}]>}, %arg1: tensor<8x16xf64> {sdy.sharding = #sdy.sharding<@dummy, [{}, {"x"}]>}, %arg2: tensor<16x8xf64> {sdy.sharding = #sdy.sharding<@dummy, [{"x"}, {}]>}) -> (tensor<8x8xf64> {sdy.sharding = #sdy.sharding<@dummy, [{}, {}]>}, tensor<8x16xf64> {sdy.sharding = #sdy.sharding<@dummy, [{}, {"x"}]>}, tensor<16x8xf64> {sdy.sharding = #sdy.sharding<@dummy, [{"x"}, {}]>}) {
///     %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@dummy, [{}, {}]>]>} : (tensor<8x16xf64>, tensor<16x8xf64>) -> tensor<8x8xf64>
///     %1 = sdy.all_reduce {"x"} %0 out_sharding=<@dummy, [{}, {}]> : tensor<8x8xf64>
///     return %1, %arg1, %arg2 : tensor<8x8xf64>, tensor<8x16xf64>, tensor<16x8xf64>
///   }
/// }
///
mlir::OwningOpRef<mlir::ModuleOp> createMatmulExample(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<stablehlo::StablehloDialect>();
  ctx.getOrLoadDialect<sdy::SdyDialect>();
  OpBuilder opBuilder(&ctx);
  auto loc = opBuilder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(opBuilder, loc));

  opBuilder.setInsertionPointToEnd(module->getBody());
  // add device mesh
  llvm::SmallVector<sdy::MeshAxisAttr> axes(2);
  axes[0] = sdy::MeshAxisAttr::get(&ctx, "x", 2);
  axes[1] = sdy::MeshAxisAttr::get(&ctx, "y", 2);
  sdy::MeshOp::create(opBuilder, loc, "dummy", ::mlir::sdy::MeshAttr::get(&ctx, axes)); 

  // define the function
  auto op0Type = mlir::RankedTensorType::get({8, 16}, opBuilder.getF64Type());
  auto op1Type = mlir::RankedTensorType::get({16, 8}, opBuilder.getF64Type());
  auto res0Type = mlir::RankedTensorType::get({8, 8}, opBuilder.getF64Type());
  auto funcType = opBuilder.getFunctionType({res0Type, op0Type, op1Type},
                                            {res0Type, op0Type, op1Type});
  auto function = func::FuncOp::create(opBuilder, loc, "main", funcType);
  auto *entry = function.addEntryBlock();
  opBuilder.setInsertionPointToStart(entry);
  auto dims =
      stablehlo::DotDimensionNumbersAttr::get(&ctx,
                                              /*lhsBatchingDimensions=*/{},
                                              /*rhsBatchingDimensions=*/{},
                                              /*lhsContractingDimensions=*/{1},
                                              /*rhsContractingDimensions=*/{0});

  auto gemmOp = stablehlo::DotGeneralOp::create(
      opBuilder, loc, res0Type, entry->getArgument(1), entry->getArgument(2),
      dims,
      /*precision_config=*/mlir::ArrayAttr{},
      /*algorithm=*/stablehlo::DotAlgorithmAttr{});
  func::ReturnOp::create(
      opBuilder, loc, ValueRange{gemmOp.getResult(), gemmOp.getLhs(), gemmOp.getRhs()});

  // Each dimension is closed: unused mesh axes remain replicated.
  auto replicatedDim = sdy::DimensionShardingAttr::get(&ctx, {}, true);
  auto splitDim = sdy::DimensionShardingAttr::get(
      &ctx, {sdy::AxisRefAttr::get(&ctx, "x")}, true);

  auto replicated = sdy::TensorShardingAttr::get(
      &ctx, "dummy", {replicatedDim, replicatedDim}, {}, {});
  auto lhsSharding = sdy::TensorShardingAttr::get(
      &ctx, "dummy", {replicatedDim, splitDim}, {}, {});
  auto rhsSharding = sdy::TensorShardingAttr::get(
      &ctx, "dummy", {splitDim, replicatedDim}, {}, {});

  // Function arguments and writeback results are ordered C, A, B.
  llvm::SmallVector<sdy::TensorShardingAttr> shardings{
      replicated, lhsSharding, rhsSharding};
  for (unsigned i = 0; i < shardings.size(); ++i) {
    function.setArgAttr(i, "sdy.sharding", shardings[i]);
    function.setResultAttr(i, "sdy.sharding", shardings[i]);
  }

  // Split K produces partial sums; returning replicated C needs an all-reduce.
  sdy::setSharding(gemmOp.getResult(), replicated);

  return module;
}

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

  auto module = createMatmulExample(ctx);
  mlir::sdy::PropagationOptions options;
  // make it easier for me to estimate the collectives
  options.enableInsertExplicitCollectives = true;

  mlir::PassManager pm(&ctx);
  mlir::sdy::addPropagationPipeline(pm, options);

  if (mlir::failed(pm.run(module.get()))) {
    return;
  }

  module->print(llvm::outs());
}
