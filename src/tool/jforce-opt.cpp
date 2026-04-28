#include "../transform/passes.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllPasses.h"
#include "stablehlo/dialect/StablehloOps.h"

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  fir::registerFIRToSCFPass();
  xla_jit::registerAllJitPasses();

  mlir::DialectRegistry registry;
  registry.insert<
    mlir::func::FuncDialect, 
    mlir::arith::ArithDialect,
    mlir::math::MathDialect,
    fir::FIROpsDialect,
    hlfir::hlfirDialect,
    mlir::omp::OpenMPDialect,
    mlir::scf::SCFDialect,
    mlir::stablehlo::StablehloDialect>();

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "XLA JIT Executing\n", registry));
}
