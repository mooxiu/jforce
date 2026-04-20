#ifndef UTILITIES_H
#define UTILITIES_H

#include "mlir/IR/Types.h"
#include <cstdint>

#ifdef ENABLE_XLA_DEBUG
#define SET_XLA_FLAG()                                                         \
  setenv("XLA_FLAGS", "--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text", 1)
#define DEBUG_PRINT(str) llvm::dbgs() << "[DEBUG]" << str << "\n"
#else
#define SET_XLA_FLAG()
#define DEBUG_PRINT(str)
#endif


bool isLiteralTy(int64_t argType);

bool isDynamicShape(mlir::Type type);

mlir::Type convertToStaticShape(mlir::Type type, llvm::ArrayRef<int64_t> shape);

std::string getMLIROperationAsString(mlir::Operation *op);

#endif
