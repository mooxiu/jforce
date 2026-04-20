#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "../support/profiler.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

using namespace mlir;

#define JIT_LITERAL_VAL_ATTR_NAME "jit.literal_val"
#define JIT_ARGS_MAPPING_ATTR_NAME "jit.args_mapping"

namespace {
struct TrimArgsPass:
  mlir::PassWrapper<TrimArgsPass, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TrimArgsPass)


  llvm::DenseMap<unsigned, unsigned> argsIndicesMapping;

  // TODO: currently store dictionary for consistency of the old code, 
  // should modified to only store new Idx array later.
  void setArgsMapToFuncAttribute(OpBuilder opBuilder, func::FuncOp funcOp) {
    llvm::SmallVector<mlir::NamedAttribute> entries;

    for (auto& entry: argsIndicesMapping) {
      std::string keyStr = std::to_string(entry.getFirst());
      entries.push_back(
        mlir::NamedAttribute(
          opBuilder.getStringAttr(keyStr), 
          opBuilder.getI64ArrayAttr(entry.getSecond())
        ));
    }
    funcOp->setAttr(JIT_ARGS_MAPPING_ATTR_NAME, opBuilder.getDictionaryAttr(entries));
    return;
  }

  void runOnOperation() override {
      PROFILE_SCOPE("trim shape args", Phase::LOWERING_EXTRA);
    auto funcOp = getOperation();
    auto context = funcOp.getContext();
    OpBuilder opBuilder(context);

    // There are 2 types of arguments we have to keep:
    // type 1: Those who are not literal type, thoese are usually allocated
    // buffer; type 2: Those who are been referred to in the function body;
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
    // Usually 2 contains 1, and we may have false negative (should have trimmed
    // but not), Reasons:
    // - But slow than incorrect
    // - If a buffer not been used, but it's already been moved to device by
    // OpenMP, we will not move extra memory
    auto argsToKeep = [&]() -> llvm::DenseSet<Value> {
      llvm::DenseSet<Value> res;
      funcOp.walk([&](Operation *op) {
        if (llvm::isa<func::FuncOp>(op) || llvm::isa<func::ReturnOp>(op)) {
          // SKIP Arguments and Return
        } else {
          auto operands = op->getOperands();
          std::for_each(operands.begin(), operands.end(), [&](Value operand) {
            if (!res.contains(operand)) {
              res.insert(operand);
            }
          });
        };
      });
      return res;
    }();

    // If result returns a different arg, it should be kept.
    // For example, arg0 in the following dot product is the return value!
    //
    //  func.func @main(%arg0: tensor<f64>, %arg1: tensor<4000000xf64>, %arg2:
    //  tensor<4000000xf64>, %arg3: tensor<i32>, %arg4: tensor<i32>) ->
    //  (tensor<f64>, tensor<4000000xf64>, tensor<4000000xf64>, tensor<i32>,
    //  tensor<i32>) { %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims =
    //  [0] x [0] : (tensor<4000000xf64>, tensor<4000000xf64>) -> tensor<f64>
    //  return %0, %arg1, %arg2, %arg3, %arg4 : tensor<f64>, tensor<4000000xf64>,
    //  tensor<4000000xf64>, tensor<i32>, tensor<i32>
    //
    auto &entryBlock = funcOp.front();
    auto rOp = mlir::dyn_cast<func::ReturnOp>(entryBlock.getTerminator());
    assert(rOp && "We should be able to get returnOp!");
    for (int i = 0; i < funcOp.getNumArguments(); i++) {
      auto argOprand = funcOp.getArgument(i);
      if (rOp.getOperand(i) != argOprand) {
        argsToKeep.insert(argOprand);
      }
    };

    // Example:
    // - old indices of args: [0, 1, 2, 3, 4, 5];
    // - indicesToKeep: [1, 3, 4];
    // - argsIndicesMapping: {1: 0, 3: 1, 4: 2}; (old idx_0 is gone, so old idx_1
    // became new idx_0; for the same reason, old idx_3 became new idx_1) Mapping
    // of Index Before Trimming: Index After Trimming
    int currNewIdx = 0;
    for (int oldIdx = 0; oldIdx < funcOp.getNumArguments(); oldIdx++) {
      auto literalAttr = funcOp.getArgAttrOfType<IntegerAttr>(oldIdx, JIT_LITERAL_VAL_ATTR_NAME);
      if (!literalAttr || argsToKeep.contains(funcOp.getArgument(oldIdx))) {
        argsIndicesMapping.insert(std::pair(oldIdx, currNewIdx));
        currNewIdx += 1;
      };
    }

    // Trim arguments whose indices not in `indicesToKeep`, we only need to do the
    // trim for the FuncOP and ReturnOp, because if they appear in other places,
    // they should be already in `indicesToKeep`.
    funcOp.walk([&](Operation *op) {
      if (llvm::isa<func::FuncOp>(op)) {
        llvm::SmallVector<Type> oldArgsTypes =
            llvm::to_vector(funcOp.getFunctionType().getInputs());
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
        auto newFuncType =
            FunctionType::get(funcOp.getContext(), newArgsTypes,
                              newArgsTypes); // Input types and output types are
                                             // the same in our case
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
    setArgsMapToFuncAttribute(opBuilder, funcOp); 
  }

};
} // namespace

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createTrimArgsPass() {
    return std::make_unique<TrimArgsPass>();
  }

  void registerTrimArgsPass() {};
}
