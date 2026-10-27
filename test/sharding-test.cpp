#include "runtime/sharding/sharding-codec.h"
#include "runtime/sharding/sharding-tuner.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <string>

using namespace mlir;

namespace {

OwningOpRef<ModuleOp> createModule(MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<stablehlo::StablehloDialect>();
  ctx.getOrLoadDialect<sdy::SdyDialect>();
  return OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&ctx)));
}

void addMesh(OpBuilder &builder, ModuleOp module, StringRef name,
             llvm::ArrayRef<MeshAxis> meshAxes) {
  builder.setInsertionPointToEnd(module.getBody());
  llvm::SmallVector<sdy::MeshAxisAttr> axes;
  for (const MeshAxis &axis : meshAxes)
    axes.push_back(sdy::MeshAxisAttr::get(builder.getContext(), axis.name,
                                          axis.size));
  sdy::MeshOp::create(builder, builder.getUnknownLoc(), name,
                      sdy::MeshAttr::get(builder.getContext(), axes));
}

func::FuncOp addFunction(OpBuilder &builder, ModuleOp module, StringRef name,
                         TypeRange inputs, TypeRange results) {
  builder.setInsertionPointToEnd(module.getBody());
  auto function = func::FuncOp::create(
      builder, builder.getUnknownLoc(), name,
      builder.getFunctionType(inputs, results));
  builder.setInsertionPointToEnd(function.addEntryBlock());
  return function;
}

stablehlo::DotGeneralOp addMatmul(OpBuilder &builder, Type resultType,
                                  Value lhs, Value rhs) {
  auto dims = stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{1},
      /*rhsContractingDimensions=*/{0});
  return stablehlo::DotGeneralOp::create(
      builder, builder.getUnknownLoc(), resultType, lhs, rhs, dims,
      /*precision_config=*/ArrayAttr{}, /*algorithm=*/stablehlo::DotAlgorithmAttr{});
}

// Construct a fresh matmul module for one strategy.
OwningOpRef<ModuleOp>
createMatmulExample(MLIRContext &ctx, llvm::ArrayRef<MeshAxis> meshAxes,
                    RankedTensorType aType, RankedTensorType bType,
                    const OperationShardingStrategy &strategy) {
  auto module = createModule(ctx);
  OpBuilder builder(&ctx);

  addMesh(builder, module.get(), "dummy", meshAxes);

  auto cType = RankedTensorType::get({aType.getDimSize(0), bType.getDimSize(1)},
                                     aType.getElementType());
  auto function = addFunction(builder, module.get(), "main",
                              {cType, aType, bType}, {cType, aType, bType});
  auto *entry = &function.getBody().front();

  // Insert operations here; addFunction leaves the builder in the entry block.
  auto gemmOp = addMatmul(builder, cType, entry->getArgument(1),
                          entry->getArgument(2));
  func::ReturnOp::create(builder, builder.getUnknownLoc(),
                         ValueRange{gemmOp.getResult(), gemmOp.getLhs(),
                                    gemmOp.getRhs()});

  // Matmul operands/results are A, B -> C; function inputs/results are C, A, B.
  auto cSharding = toShardingAttr(ctx, "dummy", strategy.results[0]);
  auto aSharding = toShardingAttr(ctx, "dummy", strategy.operands[0]);
  auto bSharding = toShardingAttr(ctx, "dummy", strategy.operands[1]);
  llvm::SmallVector<sdy::TensorShardingAttr> shardings{cSharding, aSharding,
                                                     bSharding};
  for (unsigned i = 0; i < shardings.size(); ++i) {
    function.setArgAttr(i, "sdy.sharding", shardings[i]);
    function.setResultAttr(i, "sdy.sharding", shardings[i]);
  }

  sdy::setSharding(gemmOp.getResult(), cSharding);

  return module;
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  llvm::SmallVector<MeshAxis> meshAxes;
  meshAxes.emplace_back("x", 0, 2);
  meshAxes.emplace_back("y", 1, 2);

  auto aType = RankedTensorType::get({8, 16}, Float64Type::get(&ctx));
  auto bType = RankedTensorType::get({16, 8}, Float64Type::get(&ctx));
  auto strategies = getMatmulStrategies(aType, bType, meshAxes);
  if (strategies.size() != 9) {
    llvm::errs() << "Expected nine strategies for the 2x2 mesh\n";
    return 1;
  }

  for (size_t i = 0; i < strategies.size(); ++i) {
    const auto &strategy = strategies[i];
    auto module = createMatmulExample(ctx, meshAxes, aType, bType, strategy);
    if (!module || mlir::failed(mlir::verify(module.get())))
      return 1;

    mlir::sdy::PropagationOptions options;
    options.enableInsertExplicitCollectives = true;
    mlir::PassManager pm(&ctx);
    mlir::sdy::addPropagationPipeline(pm, options);
    if (mlir::failed(pm.run(module.get())) ||
        mlir::failed(mlir::verify(module.get())))
      return 1;

    // K-sharded matmuls require reduction over the corresponding mesh axes.
    std::set<std::string> expectedAxes, actualAxes;
    for (const auto &axis : strategy.operands[0][1])
      expectedAxes.insert(axis.get().name);
    module->walk([&](mlir::sdy::AllReduceOp op) {
      for (auto axis : op.getReductionAxes())
        actualAxes.insert(axis.getName().str());
    });
    if (actualAxes != expectedAxes) {
      llvm::errs() << "Unexpected all-reduce axes for strategy " << i + 1
                   << "\n";
      module->print(llvm::errs());
      return 1;
    }

    llvm::outs() << "\n=== Strategy " << i + 1 << " of " << strategies.size()
                 << " ===\n";
    llvm::outs() << "A: " << toShardingAttr(ctx, "dummy", strategy.operands[0])
                 << "\nB: " << toShardingAttr(ctx, "dummy", strategy.operands[1])
                 << "\nC: " << toShardingAttr(ctx, "dummy", strategy.results[0])
                 << "\n";
    module->print(llvm::outs());
    llvm::outs() << "\n";
  }
  return 0;
}

