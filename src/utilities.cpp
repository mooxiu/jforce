#include "utilities.h"
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

bool isLiteralTy(int64_t argType) {
  return (bool)(argType&0x100);  
}

void logger::Log(std::string msg, logLevel level) {
  switch (level) {
  case logLevel::DEBUG:
    llvm::dbgs() << "[DEBUG] " << msg << "\n";
    break;
  case logLevel::ERROR:
    llvm::dbgs() << "[ERROR] " << msg << "\n";
    break;
  }
}

// FIXME: does not cover full situations, can be false negative.
bool isDynamicShape(mlir::Type type){
  if (auto ref = llvm::dyn_cast<fir::ReferenceType>(type)) {
    return isDynamicShape(ref.getEleTy());
  }

  if (auto box = llvm::dyn_cast<fir::BoxType>(type)) {
    return isDynamicShape(box.getEleTy());
  }

  if (auto seq = llvm::dyn_cast<fir::SequenceType>(type)){
    return seq.hasDynamicExtents();
  }

  if (auto exp = llvm::dyn_cast<hlfir::ExprType>(type)) {
    for (auto dim: exp.getShape()) {
      // TODO: is this correct usage?
      if (dim == mlir::ShapedType::kDynamic) {
        return true;
      }
    }
  }
  return false;
};

mlir::Type convertToStaticShape(mlir::Type type, llvm::ArrayRef<int64_t> shape){
  if (!isDynamicShape(type)){
    return type;
  }

  return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
    .Case<fir::ReferenceType>([&](fir::ReferenceType rType){
      return fir::ReferenceType::get(convertToStaticShape(rType.getEleTy(), shape));
    })
    .Case<fir::BoxType>([&](fir::BoxType bType){
      return fir::BoxType::get(convertToStaticShape(bType.getEleTy(), shape));
    })
    .Case<fir::SequenceType>([&](fir::SequenceType sType){
      return fir::SequenceType::get(shape, sType.getEleTy());
    })
    .Case<hlfir::ExprType>([&](hlfir::ExprType eType){
      return hlfir::ExprType::get(eType.getContext(), shape, eType.getEleTy(), eType.getPolymorphic());
    })
    .Default([&](mlir::Type t){
      return t;
    });
};


