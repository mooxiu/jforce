#ifndef UTILITIES_H
#define UTILITIES_H

#include "mlir/IR/Types.h"
#include <cstdint>

bool isLiteralTy(int64_t argType);

bool isDynamicShape(mlir::Type type);

mlir::Type convertToStaticShape(mlir::Type type, llvm::ArrayRef<int64_t> shape);

std::string getMLIROperationAsString(mlir::Operation *op);

#endif
