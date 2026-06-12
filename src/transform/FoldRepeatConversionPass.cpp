#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include <cassert>

using namespace mlir;

namespace {
struct RepeatConversionPattern : public OpRewritePattern<fir::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(fir::ConvertOp convertOp,
                                PatternRewriter &rewriter) const final {

    auto convertFromVal1 = convertOp.getOperand();
    auto convertToVal1 = convertOp.getResult();
    if (convertToVal1.getNumUses() != 1) return failure();
    if (auto declareOp = llvm::dyn_cast<fir::DeclareOp>(*convertToVal1.use_begin()->getOwner())) {
      auto declaredRes = declareOp.getResult();
      if (declaredRes.getNumUses() != 1) return failure();
      if (auto convertBackOp = llvm::dyn_cast<fir::ConvertOp>(*declaredRes.use_begin()->getOwner())) {
        if (convertBackOp.getResult().getType() == convertFromVal1.getType()) {
          rewriter.replaceAllUsesWith(convertBackOp.getResult(), convertFromVal1);
          rewriter.eraseOp(convertBackOp);
          rewriter.eraseOp(declareOp);
          rewriter.eraseOp(convertOp);
          return success();
        }
      }
    }

    return failure();
  }
};


struct FoldRepeatConversionsPass
    : public PassWrapper<FoldRepeatConversionsPass, mlir::OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldRepeatConversionsPass)

  StringRef getArgument() const override { return "jforce-fold-conversion"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RepeatConversionPattern>(ctx);
    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
    return;
  }
};
} // namespace

namespace xla_jit {
std::unique_ptr<mlir::Pass> createFoldRepeatConversionsPass() {
  return std::make_unique<FoldRepeatConversionsPass>();
}

void registerFoldRepeatConversionsPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createFoldRepeatConversionsPass(); });
};
} // namespace xla_jit
