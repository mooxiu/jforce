#ifndef UTILS_H 
#define UTILS_H 

#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

/// INFO: Jforce helper functions and types
#define JIT_SLICE_SHIFT_ATTR_NAME "jit.slice_shift"
#define JIT_LITERAL_VAL_ATTR_NAME "jit.literal_val"
#define JIT_ARG_TYPE_NAME_ATTR "jit.arg_type"
#define ALIASING_ATTRIBUTE "tf.aliasing_output"
#define JIT_ARGS_MAPPING_ATTR_NAME "jit.args_mapping"
#define JIT_OUTLINE_AFFINE_FUNC_PREFIX "outlined_affinefor_"

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

static TypeInfo inspectTypeInfoInternal(mlir::Type ty, TypeInfo& typeInfo) {
  if (auto refTy = llvm::dyn_cast<fir::ReferenceType>(ty)) {
    return inspectTypeInfoInternal(refTy.getEleTy(), typeInfo);
  }
  if (auto seqTy = llvm::dyn_cast<fir::SequenceType>(ty)) {
    typeInfo.isDynamic = seqTy.hasDynamicExtents();
    typeInfo.rank = seqTy.getShape().size();
    typeInfo.shape = seqTy.getShape();
    return inspectTypeInfoInternal(seqTy.getEleTy(), typeInfo);
  }
  if (auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(ty)) {
    typeInfo.isDynamic = !memrefTy.hasStaticShape();
    typeInfo.rank = memrefTy.getRank();
    typeInfo.shape = memrefTy.getShape();
    return inspectTypeInfoInternal(memrefTy.getElementType(), typeInfo);
  }
  if (auto tensorTy = llvm::dyn_cast<mlir::TensorType>(ty)) {
    typeInfo.isDynamic = !tensorTy.hasStaticShape();
    typeInfo.rank = tensorTy.getRank();
    typeInfo.shape = tensorTy.getShape();
    return inspectTypeInfoInternal(tensorTy.getElementType(), typeInfo);
  }
  if (ty.isIntOrIndexOrFloat()) {
    if (typeInfo.rank == 0) {
      // meaning this is not a sequence type
      typeInfo.shape = {};
      typeInfo.isDynamic = false;
    }
    typeInfo.elementTy = ty;
    return typeInfo;
  };
  llvm::errs() << "Unexpected Type: ";
  ty.print(llvm::errs());
  std::exit(EXIT_FAILURE); 
}

static TypeInfo inspectTypeInfo(mlir::Type ty) {
  TypeInfo typeInfo;
  return inspectTypeInfoInternal(ty, typeInfo);
}

#endif
