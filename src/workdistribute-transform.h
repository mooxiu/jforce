#include "kernel_pointer_interface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

void inferShape(
  MLIRContext* ctx, 
  ModuleOp moduleOp, 
  const std::vector<RegularizedTgtArg>& deviceArgs,
  llvm::DenseMap<Value, llvm::SmallVector<int>>& sliceShiftMap
); 

void optimizeSignatureForXLAAliasing(MLIRContext* context, func::FuncOp& funcOp);

func::FuncOp workdistributeToStableHLO(
  MLIRContext* context, 
  const mlir::ModuleOp& moduleOp, 
  llvm::DenseMap<Value, llvm::SmallVector<int>>& sliceShiftMap
);

llvm::DenseMap<unsigned, unsigned> trimShapeArgs(
  MLIRContext* context, 
  func::FuncOp& funcOp, 
  const std::vector<RegularizedTgtArg>& deviceArgs
);


