#ifndef UTILS_H 
#define UTILS_H 

#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdlib>
#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdlib>

/// INFO: Jforce helper functions and types
#define JIT_SLICE_SHIFT_ATTR_NAME "jit.slice_shift"
#define JIT_LITERAL_VAL_ATTR_NAME "jit.literal_val"
#define JIT_ARG_TYPE_NAME_ATTR "jit.arg_type"
#define ALIASING_ATTRIBUTE "tf.aliasing_output"
#define JIT_ARGS_MAPPING_ATTR_NAME "jit.args_mapping"
#define JIT_OUTLINE_AFFINE_FUNC_PREFIX "outlined_affinefor_"


using namespace mlir;

enum ArgType {
  SHAPE_OR_BOUND, 
  OTHER
};

struct TypeInfo {
  llvm::ArrayRef<int64_t> shape;
  int64_t rank;
  bool isDynamic;
  mlir::Type elementTy;
};


TypeInfo inspectTypeInfo(mlir::Type ty);

bool mayAccessMemory(Value val, Operation* op, fir::AliasAnalysis& aa);

bool mayWriteToMemory(Value val, Operation* op, fir::AliasAnalysis& aa);

bool mayReadFromMemory(Value val, Operation* op, fir::AliasAnalysis& aa);

#endif
