#include "../transform/passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllPasses.h"

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  xla_jit::registerAllJitPasses();

  mlir::DialectRegistry registry;
  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "XLA JIT Executing\n", registry));
}
