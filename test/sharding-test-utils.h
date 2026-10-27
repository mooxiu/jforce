#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "runtime/sharding/sharding-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace sharding_test {

// Operation pointers belong to the unchanged source module.
struct OperationChoices {
  mlir::Operation *operation;
  llvm::SmallVector<OperationShardingStrategy> strategies;
};
using Assignment = llvm::SmallVector<size_t>;

mlir::LogicalResult
collectChoices(mlir::func::FuncOp function, llvm::ArrayRef<MeshAxis> axes,
               llvm::SmallVectorImpl<OperationChoices> &choices);
std::vector<Assignment>
enumerateAssignments(llvm::ArrayRef<OperationChoices> choices, bool prune);
mlir::OwningOpRef<mlir::ModuleOp>
applyAssignment(mlir::ModuleOp source, llvm::ArrayRef<OperationChoices> choices,
                llvm::ArrayRef<size_t> assignment);
mlir::LogicalResult lowerAndCheck(mlir::ModuleOp module,
                                  llvm::ArrayRef<OperationChoices> choices,
                                  llvm::ArrayRef<size_t> assignment);
void printAssignment(llvm::StringRef example, size_t index, size_t total,
                     bool prune, mlir::ModuleOp module,
                     llvm::ArrayRef<OperationChoices> choices,
                     llvm::ArrayRef<size_t> assignment);
mlir::LogicalResult checkStrategyGenerators(mlir::MLIRContext &ctx);

} // namespace sharding_test
