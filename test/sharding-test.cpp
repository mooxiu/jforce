#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

// The helper currently has no declaration in a header.
mlir::OwningOpRef<mlir::ModuleOp>
createMatmulExample(mlir::MLIRContext &ctx);

int main() {
  mlir::MLIRContext ctx;
  auto module = createMatmulExample(ctx);

  if (!module || mlir::failed(mlir::verify(module.get())))
    return 1;

  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}

