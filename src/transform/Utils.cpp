#include "Utils.h"
#include "llvm/Support/Casting.h"

TypeInfo inspectTypeInfoInternal(mlir::Type ty, TypeInfo& typeInfo) {
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

TypeInfo inspectTypeInfo(mlir::Type ty) {
  TypeInfo typeInfo;
  return inspectTypeInfoInternal(ty, typeInfo);
}

