#include "sharding-test-utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "runtime/sharding/sharding-codec.h"
#include "runtime/sharding/sharding-tuner.h"
#include "sharding-examples.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

using namespace mlir;

namespace sharding_test {
namespace {

constexpr llvm::StringLiteral shardingName = "sdy.sharding";
constexpr llvm::StringLiteral operationId = "sharding_test.id";

bool isElementwise(Operation *op) {
  return isa<stablehlo::AddOp, stablehlo::MulOp, stablehlo::SubtractOp>(op);
}

const TensorSharding *producerSharding(Value value,
                                  llvm::ArrayRef<OperationChoices> choices,
                                  llvm::ArrayRef<size_t> assignment,
                                  size_t consumer) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return nullptr;
  for (size_t i = 0; i < consumer; ++i)
    if (choices[i].operation == result.getOwner())
      return &choices[i]
                  .strategies[assignment[i]]
                  .results[result.getResultNumber()];
  return nullptr; // Function arguments have no producer choice.
}

void enumerateFrom(llvm::ArrayRef<OperationChoices> choices, bool prune,
                   size_t next, Assignment &current,
                   std::vector<Assignment> &assignments) {
  if (next == choices.size()) {
    assignments.push_back(current);
    return;
  }
  auto *op = choices[next].operation;
  for (size_t i = 0; i < choices[next].strategies.size(); ++i) {
    if (prune && isElementwise(op) &&
        !keepEleBinaryOpStrategy(
            choices[next].strategies[i],
            producerSharding(op->getOperand(0), choices, current, next),
            producerSharding(op->getOperand(1), choices, current, next)))
      continue;
    current[next] = i;
    enumerateFrom(choices, prune, next + 1, current, assignments);
  }
}

Value reshardForUse(OpBuilder &builder, Value value,
                    sdy::TensorShardingAttr target) {
  if (sdy::getSharding(value) == target)
    return value;
  return sdy::ReshardOp::create(builder, builder.getUnknownLoc(), value, target)
      .getResult();
}

bool matchesStrategy(Operation *op, const OpStrategy &strategy) {
  auto matches = [&](Value value, const TensorSharding &sharding) {
    auto actual = sdy::getSharding(value);
    auto expected = toShardingAttr(*op->getContext(), meshName, sharding);
    // A matmul's intermediate result may additionally carry unreduced axes.
    return actual && actual.getDimShardings() == expected.getDimShardings();
  };
  if (op->getNumOperands() != strategy.operands.size() ||
      op->getNumResults() != strategy.results.size())
    return false;
  for (unsigned i = 0; i < op->getNumOperands(); ++i)
    if (!matches(op->getOperand(i), strategy.operands[i]))
      return false;
  for (unsigned i = 0; i < op->getNumResults(); ++i)
    if (!matches(op->getResult(i), strategy.results[i]))
      return false;
  return true;
}

} // namespace

LogicalResult collectChoices(func::FuncOp function,
                             llvm::ArrayRef<MeshAxis> axes,
                             llvm::SmallVectorImpl<OperationChoices> &choices) {
  if (!function || function.isExternal() ||
      function.getBody().getBlocks().size() != 1) {
    llvm::errs() << "The example must define a single-block @main function\n";
    return failure();
  }
  for (Type type : function.getArgumentTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape()) {
      function.emitError("test arguments must be static ranked tensors");
      return failure();
    }
  }
  for (Operation &operation : function.getBody().front()) {
    auto *op = &operation;
    if (isa<func::ReturnOp>(op))
      continue;
    llvm::SmallVector<OpStrategy> strategies;
    if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(op)) {
      auto dims = dot.getDotDimensionNumbers();
      if (!dims.getLhsBatchingDimensions().empty() ||
          !dims.getRhsBatchingDimensions().empty() ||
          dims.getLhsContractingDimensions().size() != 1 ||
          dims.getRhsContractingDimensions().size() != 1 ||
          dims.getLhsContractingDimensions()[0] != 1 ||
          dims.getRhsContractingDimensions()[0] != 0) {
        op->emitError("only ordinary 2D matmul strategies are supported");
        return failure();
      }
      strategies = getMatmulStrategies(
          dyn_cast<RankedTensorType>(dot.getLhs().getType()),
          dyn_cast<RankedTensorType>(dot.getRhs().getType()), axes);
    } else if (isElementwise(op)) {
      strategies = getEleBinaryOpStrategies(
          dyn_cast<RankedTensorType>(op->getOperand(0).getType()),
          dyn_cast<RankedTensorType>(op->getOperand(1).getType()), axes);
    } else {
      op->emitError("no sharding strategy generator for this operation");
      return failure();
    }
    if (strategies.empty()) {
      op->emitError("no legal strategies for these shapes and mesh");
      return failure();
    }
    choices.push_back({op, std::move(strategies)});
  }
  if (choices.empty()) {
    function.emitError("example has no computation operations");
    return failure();
  }
  return success();
}

std::vector<Assignment>
enumerateAssignments(llvm::ArrayRef<OperationChoices> choices, bool prune) {
  std::vector<Assignment> assignments;
  Assignment current(choices.size(), 0);
  enumerateFrom(choices, prune, 0, current, assignments);
  return assignments;
}

double countAssignments(llvm::ArrayRef<OperationChoices> choices) {
  double count = 1;
  for (const auto &choice : choices) {
    count *= choice.strategies.size();
  }
  return count;
}

OwningOpRef<ModuleOp> applyAssignment(ModuleOp source,
                                      llvm::ArrayRef<OperationChoices> choices,
                                      llvm::ArrayRef<size_t> assignment) {
  auto module = OwningOpRef<ModuleOp>(source.clone());
  auto function = module->lookupSymbol<func::FuncOp>("main");
  auto &entry = function.getBody().front();
  llvm::SmallVector<Operation *> operations;
  for (Operation &op : entry)
    if (!isa<func::ReturnOp>(&op))
      operations.push_back(&op);
  OpBuilder builder(module.get().getContext());
  llvm::SmallVector<bool> assigned(function.getNumArguments(), false);
  for (unsigned i = 0; i < assigned.size(); ++i)
    assigned[i] = bool(function.getArgAttr(i, shardingName));

  for (size_t i = 0; i < choices.size(); ++i) {
    auto *op = operations[i];
    const auto &strategy = choices[i].strategies[assignment[i]];
    builder.setInsertionPoint(op);
    for (unsigned j = 0; j < op->getNumOperands(); ++j) {
      Value value = op->getOperand(j);
      auto target =
          toShardingAttr(*op->getContext(), meshName, strategy.operands[j]);
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        auto index = argument.getArgNumber();
        if (!assigned[index]) {
          function.setArgAttr(index, shardingName, target);
          assigned[index] = true; // Boundary sharding follows the first consumer.
        }
      }
      op->setOperand(j, reshardForUse(builder, value, target));
    }
    for (unsigned j = 0; j < op->getNumResults(); ++j)
      sdy::setSharding(
          op->getResult(j),
          toShardingAttr(*op->getContext(), meshName, strategy.results[j]));
    op->setAttr(operationId, builder.getI64IntegerAttr(i));
  }

  // Unused arguments are replicated. Returns may have any order or sharding.
  for (unsigned i = 0; i < assigned.size(); ++i)
    if (!assigned[i]) {
      auto type = cast<RankedTensorType>(function.getArgument(i).getType());
      function.setArgAttr(i, shardingName,
                          toShardingAttr(*function.getContext(), meshName,
                                         TensorSharding(type.getRank())));
    }
  auto terminator = cast<func::ReturnOp>(entry.getTerminator());
  builder.setInsertionPoint(terminator);
  for (unsigned i = 0; i < terminator.getNumOperands(); ++i) {
    Value value = terminator.getOperand(i);
    auto target =
        function.getResultAttrOfType<sdy::TensorShardingAttr>(i, shardingName);
    if (target)
      terminator->setOperand(i, reshardForUse(builder, value, target));
    else
      function.setResultAttr(i, shardingName, sdy::getSharding(value));
  }
  return module;
}

LogicalResult lowerAndCheck(ModuleOp module,
                            llvm::ArrayRef<OperationChoices> choices,
                            llvm::ArrayRef<size_t> assignment) {
  if (failed(verify(module)))
    return failure();
  sdy::PropagationOptions options;
  options.enableInsertExplicitCollectives = true;
  PassManager pm(module.getContext());
  sdy::addPropagationPipeline(pm, options);
  if (failed(pm.run(module)) || failed(verify(module)))
    return failure();

  bool valid = true;
  llvm::SmallVector<bool> seen(choices.size(), false);
  llvm::SmallVector<Operation *> labelled;
  module.walk([&](Operation *op) {
    auto id = op->getAttrOfType<IntegerAttr>(operationId);
    if (!id)
      return;
    auto index = id.getInt();
    if (index < 0 || size_t(index) >= choices.size() || seen[index]) {
      op->emitError("invalid or duplicate test operation identity");
      valid = false;
      return;
    }
    seen[index] = true;
    labelled.push_back(op);
    if (op->getName() != choices[index].operation->getName() ||
        !matchesStrategy(op, choices[index].strategies[assignment[index]])) {
      op->emitError("selected sharding strategy changed during lowering");
      valid = false;
    }
  });
  if (!valid || labelled.size() != choices.size()) {
    module.emitError("not all selected operations/shardings survived lowering");
    return failure();
  }
  for (auto *op : labelled)
    op->removeAttr(
        operationId); // The identity is only needed for verification.
  return success();
}

void printAssignment(llvm::StringRef example, size_t index, size_t total,
                     bool prune, ModuleOp module,
                     llvm::ArrayRef<OperationChoices> choices,
                     llvm::ArrayRef<size_t> assignment) {
  llvm::outs() << "\n=== " << example << ": Combination " << index + 1 << " of "
               << total << (prune ? " (pruned)" : " (exhaustive)") << " ===\n";
  for (size_t i = 0; i < choices.size(); ++i) {
    const auto &strategy = choices[i].strategies[assignment[i]];
    llvm::outs() << "op" << i << ": " << choices[i].operation->getName()
                 << " strategy " << assignment[i] + 1 << "\n";
    for (size_t j = 0; j < strategy.operands.size(); ++j)
      llvm::outs() << "  operand " << j << ": "
                   << toShardingAttr(*module.getContext(), meshName,
                                     strategy.operands[j])
                   << "\n";
    for (size_t j = 0; j < strategy.results.size(); ++j)
      llvm::outs() << "  result " << j << ": "
                   << toShardingAttr(*module.getContext(), meshName,
                                     strategy.results[j])
                   << "\n";
  }
  module.print(llvm::outs());
  llvm::outs() << "\n";
}

LogicalResult checkStrategyGenerators(MLIRContext &ctx) {
  llvm::SmallVector<MeshAxis> axes{{"x", 0, 2}, {"y", 1, 2}};
  auto a = RankedTensorType::get({8, 16}, Float64Type::get(&ctx));
  auto b = RankedTensorType::get({16, 8}, a.getElementType());
  auto scalar = RankedTensorType::get({}, a.getElementType());
  auto uneven = RankedTensorType::get({6, 4}, a.getElementType());
  if (getMatmulStrategies(a, b, axes).size() != 9 ||
      getEleBinaryOpStrategies(a, a, axes).size() != 9 ||
      getEleBinaryOpStrategies(scalar, scalar, axes).size() != 1 ||
      getEleBinaryOpStrategies(uneven, uneven, axes).size() != 8 ||
      !getEleBinaryOpStrategies(a, b, axes).empty()) {
    llvm::errs() << "Strategy generator checks failed\n";
    return failure();
  }
  TensorSharding lhs(2), rhs(2);
  lhs[0].push_back(axes[0]);
  lhs[1].push_back(axes[1]);
  rhs[0].push_back(axes[0]);
  size_t kept = 0;
  for (const auto &candidate : getEleBinaryOpStrategies(a, a, axes))
    if (keepEleBinaryOpStrategy(candidate, &lhs, &rhs)) {
      ++kept;
      if (toStr(candidate.results) != toStr(TensorShardings{rhs})) {
        llvm::errs() << "Elementwise conflict-dimension pruning failed\n";
        return failure();
      }
    }
  if (kept != 1) {
    llvm::errs() << "Expected one merged elementwise strategy\n";
    return failure();
  }
  return success();
}

LogicalResult checkTuner(llvm::StringRef example, func::FuncOp function,
                         llvm::ArrayRef<MeshAxis> axes) {
  using Clock = std::chrono::steady_clock;
  auto elapsedMs = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
  };

  Graph graph = buildGraph(function);
  auto ilpStart = Clock::now();
  auto plan = graph.tune(axes);
  double ilpMs = elapsedMs(ilpStart);
  for (Operation *op : graph.opSet) {
    if (!plan.contains(op)) {
      op->emitError("the ILP plan has no strategy for this operation");
      return failure();
    }
  }
  double ilpCost = getPlanCost(graph, plan);

  // Same strategies and cost model as the ILP, evaluated on every combination.
  llvm::SmallVector<OperationChoices> choices;
  if (failed(collectChoices(function, axes, choices))) {
    return failure();
  }
  auto search = [&](bool prune) {
    auto start = Clock::now();
    auto assignments = enumerateAssignments(choices, prune);
    double best = std::numeric_limits<double>::infinity();
    for (const auto &assignment : assignments) {
      ShardingPlan candidate;
      for (size_t i = 0; i < choices.size(); ++i) {
        candidate[choices[i].operation] =
            choices[i].strategies[assignment[i]];
      }
      best = std::min(best, getPlanCost(graph, candidate));
    }
    llvm::outs() << llvm::formatv("  {0,-10} cost {1,6} bytes, {2,6} plans, "
                                  "{3,8:f2} ms\n",
                                  prune ? "pruned" : "exhaustive", best,
                                  assignments.size(), elapsedMs(start));
    return best;
  };

  llvm::outs() << "=== " << example << " ===\n";
  llvm::outs() << llvm::formatv("  {0,-10} cost {1,6} bytes, {2,6} plans, "
                                "{3,8:f2} ms\n",
                                "ILP", ilpCost, 1, ilpMs);
  double total = countAssignments(choices);
  bool enumerable = total <= maxEnumeratedAssignments;
  double exhaustiveCost = ilpCost;
  if (enumerable) {
    exhaustiveCost = search(/*prune=*/false);
    search(/*prune=*/true);
  } else {
    llvm::outs() << llvm::formatv("  exhaustive and pruned search skipped: "
                                  "{0:e2} plans; optimality not verified\n",
                                  total);
  }

  // The chosen strategy of every op, in program order.
  llvm::outs() << "  ILP plan:\n";
  for (size_t i = 0; i < choices.size(); ++i) {
    Operation *op = choices[i].operation;
    const auto &strategy = plan.at(op);
    llvm::outs() << "    op" << i << ": " << op->getName() << "\n";
    for (size_t j = 0; j < strategy.operands.size(); ++j) {
      llvm::outs() << "      operand " << j << ": "
                   << toShardingAttr(*op->getContext(), meshName,
                                     strategy.operands[j])
                   << "\n";
    }
    for (size_t j = 0; j < strategy.results.size(); ++j) {
      llvm::outs() << "      result " << j << ": "
                   << toShardingAttr(*op->getContext(), meshName,
                                     strategy.results[j])
                   << "\n";
    }
  }
  // The ILP must reach the exhaustive optimum; pruning can only match or lose.
  if (enumerable && std::abs(ilpCost - exhaustiveCost) >
      1e-6 * std::max(1.0, exhaustiveCost)) {
    llvm::errs() << example << ": the ILP plan is not optimal\n";
    return failure();
  }
  return success();
}

} // namespace sharding_test
