#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "runtime/sharding/sharding-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace sharding_test {

inline constexpr llvm::StringLiteral meshName = "mesh";

// Builders create unannotated IR. The search applies layouts separately.
mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               llvm::ArrayRef<MeshAxis> axes);
mlir::func::FuncOp addFunction(mlir::OpBuilder &builder, mlir::ModuleOp module,
                               mlir::TypeRange inputs, mlir::TypeRange results);
mlir::Value addMatmul(mlir::OpBuilder &builder, mlir::Value lhs,
                      mlir::Value rhs);

struct Example {
  llvm::StringRef name;
  llvm::StringRef description;
  mlir::OwningOpRef<mlir::ModuleOp> (*build)(mlir::MLIRContext &,
                                             llvm::ArrayRef<MeshAxis>);
};

llvm::ArrayRef<Example> getExamples();

} // namespace sharding_test
