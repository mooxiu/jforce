#include "utilities.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>

using namespace mlir;

#define ALIASING_ATTRIBUTE "tf.aliasing_output" 

// Ref: https://openxla.org/xla/aliasing
// XLA code: `xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc`, 
// function: `ConvertToHloModule::RunOnFunction`
void optimizeSignatureForXLAAliasing(MLIRContext* context, func::FuncOp& funcOp) {
  OpBuilder opBuilder(context);
  for (int i = 0; i < funcOp.getNumArguments(); i++) {
    // do not donate literal buffers, as we will not reuse
    if (funcOp.getArgument(i).getType().isSignlessIntOrIndexOrFloat()) {
      continue;
    }
    funcOp.setArgAttr(i, ALIASING_ATTRIBUTE, opBuilder.getI64IntegerAttr(i));
  }
  return;
}


/// Some arguments are there just meant to be shape meta data, need to drop them for better performance.
/// Return a map mapping original Index -> new Index;
llvm::DenseMap<unsigned, unsigned> trimShapeArgs(MLIRContext* context, func::FuncOp& funcOp, int64_t* ArgTypes) {
  OpBuilder opBuilder(context);


  // There are 2 types of arguments we have to keep:
  // type 1: Those who are not literal type, thoese are usually allocated buffer;
  // type 2: Those who are been referred to in the function body;
  //
  // For example:
  // ```
  // func(arg1, arg2, arg3) {
  //   arg4 = add arg2, arg3
  //   return arg1, arg2, arg3
  // }
  // ```
  // Above case, arg2 and arg3 are been used, arg1 not.
  // Therefore, it can be transformed into:
  // ```
  // func (arg2, arg3) {
  //   arg4 = add arg2, arg3
  //   return arg2, arg3
  // }
  // ```
  // Usually 2 contains 1, and we may have false negative (should have trimmed but not), Reasons:
  // - But slow than incorrect
  // - If a buffer not been used, but it's already been moved to device by OpenMP, we will not move extra memory
  auto argsBeenUsed = [&]() -> llvm::DenseSet<Value> {
    llvm::DenseSet<Value> res;
    funcOp.walk([&](Operation* op){
      if (llvm::isa<func::FuncOp>(op) || llvm::isa<func::ReturnOp>(op)){
        // SKIP Arguments and Return
      } else {
        auto operands = op->getOperands();
        std::for_each(operands.begin(), operands.end(), [&](Value operand){
          if (!res.contains(operand)) {
            res.insert(operand);
          }
        });
      };
    });
    return res;
  }();

  // Example: 
  // - old indices of args: [0, 1, 2, 3, 4, 5];
  // - indicesToKeep: [1, 3, 4];
  // - argsIndicesMapping: {1: 0, 3: 1, 4: 2}; (old idx_0 is gone, so old idx_1 became new idx_0; for the same reason, old idx_3 became new idx_1)
  // Mapping of Index Before Trimming: Index After Trimming
  llvm::DenseMap<unsigned, unsigned> argsIndicesMapping;
  int currNewIdx = 0;
  for (int oldIdx = 0; oldIdx < funcOp.getNumArguments(); oldIdx++) {
    if (!isLiteralTy(ArgTypes[oldIdx]) || argsBeenUsed.contains(funcOp.getArgument(oldIdx))) {
      argsIndicesMapping.insert(std::pair(oldIdx, currNewIdx));
      currNewIdx += 1;
    };
  }

  // Trim arguments whose indices not in `indicesToKeep`, we only need to do the trim for the FuncOP and ReturnOp,
  // because if they appear in other places, they should be already in `indicesToKeep`.
  funcOp.walk([&](Operation * op){
    if (llvm::isa<func::FuncOp>(op)){
      llvm::SmallVector<Type> oldArgsTypes = llvm::to_vector(funcOp.getFunctionType().getInputs());
      llvm::SmallVector<Type> newArgsTypes;
      newArgsTypes.reserve(oldArgsTypes.size());
      for (int i = 0; i < funcOp.getNumArguments(); i++) {
        if (argsIndicesMapping.contains(i)) {
          newArgsTypes.push_back(oldArgsTypes[i]); 
        }      
      }

      // set entry block type
      Block &entryBlock = funcOp.front();
      for (int i = entryBlock.getNumArguments() - 1; i >= 0; --i) {
        if (!argsIndicesMapping.contains(i)) {
          entryBlock.eraseArgument(i);
        }
      }

      // set func type
      auto newFuncType = FunctionType::get(funcOp.getContext(), newArgsTypes, newArgsTypes); // Input types and output types are the same in our case
      funcOp.setType(newFuncType);
    } else if (auto retOp = llvm::dyn_cast<func::ReturnOp>(op)) {
      opBuilder.setInsertionPoint(retOp);

      llvm::SmallVector<Value> retOperands; 
      for (int i = 0; i < retOp.getNumOperands(); i++) {
        if (argsIndicesMapping.contains(i)) {
          retOperands.push_back(retOp.getOperand(i));
        }
      }
      func::ReturnOp::create(opBuilder, funcOp.getLoc(), retOperands);
      retOp.erase();   
    } else {
      // DO NOTHING
    };
  });
  
  return argsIndicesMapping;  
}


