#include "Utils.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"



llvm::SmallVector<mlir::MemoryEffects::EffectInstance> getMemoryEffects(Operation* op) {
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  auto memEffectInterface = llvm::dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectInterface) return effects;

  memEffectInterface.getEffects(effects);
  return effects;
}

bool mayAccessMemory(Value val, Operation* op, fir::AliasAnalysis& aa) {
  auto effects = getMemoryEffects(op);
  for (const auto& effect: effects) {
    if (effect.getValue()) {
      if (aa.alias(effect.getValue(), val).isMay()) return true;
    }
  }
  return false;
}

bool mayWriteToMemory(Value val, Operation* op, fir::AliasAnalysis& aa) {
  auto effects = getMemoryEffects(op);
  for (const auto& effect: effects) {
    if (isa<MemoryEffects::Write>(effect.getEffect()) && effect.getValue()) {
      if (aa.alias(effect.getValue(), val).isMay()) return true;
    }
  }
  return false;
}

bool mayReadFromMemory(Value val, Operation* op, fir::AliasAnalysis& aa) {
  auto effects = getMemoryEffects(op);
  for (const auto& effect: effects) {
    if (isa<MemoryEffects::Read>(effect.getEffect()) && effect.getValue()) {
      if (aa.alias(effect.getValue(), val).isMay()) return true;
    }
  }
  return false;
}



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
  if (auto boxTy = llvm::dyn_cast<fir::BoxType>(ty)) {
    return inspectTypeInfoInternal(boxTy.getElementType(), typeInfo);
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

