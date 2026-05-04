#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

/// Swap side of predicate
static mlir::arith::CmpIPredicate swapPredicate(mlir::arith::CmpIPredicate pred) {
  switch (pred) {
  case mlir::arith::CmpIPredicate::eq:
  case mlir::arith::CmpIPredicate::ne:
    return pred;
  case mlir::arith::CmpIPredicate::slt:
    return mlir::arith::CmpIPredicate::sgt;
  case mlir::arith::CmpIPredicate::sle:
    return mlir::arith::CmpIPredicate::sge;
  case mlir::arith::CmpIPredicate::sgt:
    return mlir::arith::CmpIPredicate::slt;
  case mlir::arith::CmpIPredicate::sge:
    return mlir::arith::CmpIPredicate::sle;
  case mlir::arith::CmpIPredicate::ult:
    return mlir::arith::CmpIPredicate::ugt;
  case mlir::arith::CmpIPredicate::ule:
    return mlir::arith::CmpIPredicate::uge;
  case mlir::arith::CmpIPredicate::ugt:
    return mlir::arith::CmpIPredicate::ult;
  case mlir::arith::CmpIPredicate::uge:
    return mlir::arith::CmpIPredicate::ule;
  }
  llvm_unreachable("unknown cmpi predicate kind");
} 

///

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
  if (ty.isIntOrIndexOrFloat()) {
    if (typeInfo.rank == 0) {
      // meaning this is not a sequence type
      typeInfo.shape = {};
      typeInfo.isDynamic = false;
    }
    typeInfo.elementTy = ty;
    return typeInfo;
  };
  llvm::errs() << "Unexpected Type!\n";
  std::exit(EXIT_FAILURE); 
}

static TypeInfo inspectTypeInfo(mlir::Type ty) {
  TypeInfo typeInfo;
  return inspectTypeInfoInternal(ty, typeInfo);
}

