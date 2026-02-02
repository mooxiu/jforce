#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>
#include <vector>

enum class logLevel { DEBUG, ERROR };

class logger {
public:
  static void Log(std::string msg, logLevel level);
};

bool isLiteralTy(int64_t argType);

bool isDynamicShape(mlir::Type type);

mlir::Type convertToStaticShape(mlir::Type type, llvm::ArrayRef<int64_t> shape);

