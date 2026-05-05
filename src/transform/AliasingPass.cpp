#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "../support/profiler.h"
#include "mlir/Pass/PassRegistry.h"
#include "Passes.h"
#include <memory>

using namespace mlir; 

#define ALIASING_ATTRIBUTE "tf.aliasing_output"

namespace {
struct AliasingPass : 
  public PassWrapper<AliasingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AliasingPass)

  StringRef getArgument() const override { 
    return "jforce-aliasing"; 
  }


  // Ref: https://openxla.org/xla/aliasing
  // XLA code: `xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc`,
  // function: `ConvertToHloModule::RunOnFunction`
  void runOnOperation() override {
    PROFILE_SCOPE("memory donate", Phase::LOWERING_EXTRA);
    auto funcOp = getOperation();
    auto context = funcOp.getContext();
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

};
} // namespace

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createAliasingPass() {
    return std::make_unique<AliasingPass>();
  }

  void registerAliasingPass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createAliasingPass();});
  };
}
