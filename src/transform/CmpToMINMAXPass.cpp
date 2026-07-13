//INFO: AffineCFG can only recognize single operation as reduction, thus we need to do preprocess first.

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct SelectCmpFToMinMax : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    if (!selectOp->getParentOfType<affine::AffineForOp>() &&
        !selectOp->getParentOfType<affine::AffineParallelOp>())
      return failure();

    auto cmpOp = selectOp.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!cmpOp) return failure();

    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    bool selectLhsRhs = selectOp.getTrueValue() == lhs && selectOp.getFalseValue() == rhs;
    bool selectRhsLhs = selectOp.getTrueValue() == rhs && selectOp.getFalseValue() == lhs;
    if (!selectLhsRhs && !selectRhsLhs) return failure();

    using Pred = arith::CmpFPredicate;
    Pred pred = cmpOp.getPredicate();

    bool isMax = false;
    bool isMin = false;

    switch (pred) {
    case Pred::OGT:
    case Pred::OGE:
      // select(lhs > rhs, lhs, rhs) => max(lhs, rhs)
      // select(lhs > rhs, rhs, lhs) => min(lhs, rhs)
      isMax = selectLhsRhs;
      isMin = selectRhsLhs;
      break;
    case Pred::OLT:
    case Pred::OLE:
      // select(lhs < rhs, lhs, rhs) => min(lhs, rhs)
      // select(lhs < rhs, rhs, lhs) => max(lhs, rhs)
      isMin = selectLhsRhs;
      isMax = selectRhsLhs;
      break;
    default:
      return failure();
    }

    if (isMax) {
      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(selectOp, lhs, rhs);
      return success();
    }

    if (isMin) {
      rewriter.replaceOpWithNewOp<arith::MinimumFOp>(selectOp, lhs, rhs);
      return success();
    }

    return failure();
  }
};

struct RecognizeMinMaxPass
    : public mlir::PassWrapper<RecognizeMinMaxPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-loop-sink"; }

  void runOnOperation() override {
    auto ctx = getOperation()->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<SelectCmpFToMinMax>(ctx);
    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
      return;
    }
    getOperation()->dump();
    return;
  }
};
};


namespace xla_jit {
std::unique_ptr<mlir::Pass> createRecognizeMinMaxPass() {
  return std::make_unique<RecognizeMinMaxPass>();
}

void registerRecognizeMinMaxPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createRecognizeMinMaxPass(); });
};
} // namespace xla_jit
