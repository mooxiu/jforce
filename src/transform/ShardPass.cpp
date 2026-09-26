#include "../support/utilities.h"
#include "../support/profiler.h"
#include "Passes.h"
#include "Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {
struct ShardPass : public mlir::PassWrapper<ShardPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShardPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<stablehlo::StablehloDialect>();
  }

  StringRef getArgument() const override {
    return "jforce-shard-kernel";
  }

  void runOnOperation() override {
  }
};
}; // namespace

namespace xla_jit {
std::unique_ptr<mlir::Pass> createShardPass() {
  return std::make_unique<ShardPass>();
}

void registerShardPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createShardPass(); });
};
} // namespace xla_jit
