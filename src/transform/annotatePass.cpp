#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "../support/profiler.h"
#include "llvm/Support/Casting.h"

using namespace mlir; 

#define JIT_COMPUTE_ARG_ATTR_NAME "jit.compute_arg"

namespace {
struct AnnotatePass: 
  public PassWrapper<AnnotatePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotatePass)
  
  StringRef getArgument() const override { 
    return "jforce-annotate"; 
  }


  void runOnOperation() override {
    PROFILE_SCOPE("annotate compute args", Phase::LOWERING_EXTRA);
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());

    for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
      auto arg = funcOp.getArgument(i);
      bool isCompute = false;
      
      for (auto* user: arg.getUsers()) {
        if (llvm::isa<hlfir::DeclareOp>(user) || llvm::isa<fir::DeclareOp>(user)) {
          isCompute = true;
        } 
      }

      if (isCompute) {
        funcOp.setArgAttr(i, JIT_COMPUTE_ARG_ATTR_NAME, opBuilder.getUnitAttr());
      }
    }
  }
};
} // namespace

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createAnnotatePass() {
    return std::make_unique<AnnotatePass>();
  }

  void registerAnnotatePass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createAnnotatePass();});
  };
}
