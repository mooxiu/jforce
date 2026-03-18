#ifndef UTILITIES_H 
#define UTILITIES_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <cstdlib>


enum class logLevel { DEBUG, ERROR };

class logger {
public:
  static void Log(std::string msg, logLevel level);
};

bool isLiteralTy(int64_t argType);
bool isMappedTy(int64_t argType);
bool isPartOfStructTy(int64_t argType); 
bool isPointerAndPointeeTy(int64_t argType); 

bool isDynamicShape(mlir::Type type);

mlir::Type convertToStaticShape(mlir::Type type, llvm::ArrayRef<int64_t> shape);

std::string getMLIROperationAsString(mlir::Operation* op);

#endif

