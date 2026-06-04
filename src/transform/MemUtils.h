#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace {

inline bool isOperationPossiblelyWriteToAddr(Operation* op, Value addr) {
  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memInterface) {
    return llvm::is_contained(op->getOperands(), addr);
  }
  
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memInterface.getEffects(effects);
  for (const auto &effect : effects) {
    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      Value effectValue = effect.getValue();
      if (effectValue == addr || effectValue == nullptr) {
        return true;
      }
    }
  }
  return false;
}

inline bool isOperationPossiblelyReadFromAddr(Operation* op, Value addr) {
  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memInterface) {
    return llvm::is_contained(op->getOperands(), addr);
  }
  
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memInterface.getEffects(effects);
  for (const auto &effect : effects) {
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      Value effectValue = effect.getValue();
      if (effectValue == addr || effectValue == nullptr) {
        return true;
      }
    }
  }
  return false;
}
};
