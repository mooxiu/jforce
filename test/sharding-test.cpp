#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "llvm/Support/raw_ostream.h"

// The helper currently has no declaration in a header.
mlir::OwningOpRef<mlir::ModuleOp>
createMatmulExample(mlir::MLIRContext &ctx);

int main() {
  mlir::MLIRContext ctx;
  auto module = createMatmulExample(ctx);

  if (!module || mlir::failed(mlir::verify(module.get())))
    return 1;

  mlir::sdy::PropagationOptions options;
  options.enableInsertExplicitCollectives = true;
  mlir::PassManager pm(&ctx);
  mlir::sdy::addPropagationPipeline(pm, options);
  if (mlir::failed(pm.run(module.get())) ||
      mlir::failed(mlir::verify(module.get())))
    return 1;

  bool hasAllReduce = false;
  module->walk([&](mlir::sdy::AllReduceOp) { hasAllReduce = true; });
  if (!hasAllReduce) {
    llvm::errs() << "Expected an all-reduce for the split-K matmul\n";
    return 1;
  }

  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}

