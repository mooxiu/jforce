#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

struct ReplaceFIRNoReassoc: OpRewritePattern<fir::NoReassocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(fir::NoReassocOp noReassocOp,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceAllUsesWith(noReassocOp.getRes(), noReassocOp.getVal());
    rewriter.eraseOp(noReassocOp);
    return success();
  }
};


// cases like %182 = fir.convert %29 : (!fir.ref<f64>) -> memref<f64>, might be in a if-else condition or in a loop
struct HositFIRConvertOps: OpRewritePattern<fir::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(fir::ConvertOp convertOp,
                                PatternRewriter &rewriter) const final {
    auto definedOp = convertOp.getOperand().getDefiningOp();
    if (definedOp->getParentOp() != convertOp->getParentOp()) {
      rewriter.moveOpBefore(convertOp, convertOp->getParentOp());
      return success();
    }
    return failure();
  }
};

// Before:
//  1. fir.convert index -> i32/64 
//  2. fir.i32 -> i64
//  3. fir.i64 -> i32

//
// After:
//  1. arith.index_case index -> i32/i64
//  2. arith.extsi i32 -> i64
//  3. arith.trunci i64 -> i32
struct ReplaceFIRConvertOps : OpRewritePattern<fir::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(fir::ConvertOp convertOp,
                                PatternRewriter &rewriter) const final {
    auto fromVal = convertOp.getOperand();
    auto fromValType = fromVal.getType();
    auto toVal = convertOp.getResult();
    auto toValType = toVal.getType();

    if (!(fromValType.isIntOrIndex() && toValType.isIntOrIndex())) {
      return failure();
    }

    mlir::IntegerAttr attr;
    if (matchPattern(fromVal, m_Constant(&attr))) {
      auto constVal = attr.getInt();
      if (toValType.isIndex()) {
        // arith::ConstantIndexOp::create(rewriter, convertOp.getLoc(),
        // constVal);
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(convertOp,
                                                            constVal);
      } else if (toValType.isInteger()) {
        rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(
            convertOp, constVal, toValType.getIntOrFloatBitWidth());
      } else {
        return failure();
      }
      return success();
    }

    Operation *castOp;
    if (fromValType.isIndex() && toValType.isInteger()) {
      rewriter.replaceOpWithNewOp<arith::IndexCastOp>(convertOp, toValType,
                                                      fromVal);
      return success();
    } else if (fromValType.isInteger() && toValType.isInteger()) {
      if (fromValType.getIntOrFloatBitWidth() <
          toValType.getIntOrFloatBitWidth()) {
        rewriter.replaceOpWithNewOp<arith::ExtSIOp>(convertOp, toValType,
                                                    fromVal);
        return success();
      } else {
        rewriter.replaceOpWithNewOp<arith::TruncIOp>(convertOp, toValType,
                                                     fromVal);
        return success();
      }
    }
    return failure();
  }
};


struct CleanFIROpsPass
    : public mlir::PassWrapper<CleanFIROpsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-clean-fir-op"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext* ctx = getOperation()->getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ReplaceFIRConvertOps>(ctx);
    patterns.add<ReplaceFIRNoReassoc>(ctx);
    patterns.add<HositFIRConvertOps>(ctx);
    GreedyRewriteConfig config;
    config.enableFolding();
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    
    if (failed(applyPatternsGreedily(funcOp, frozenPatterns, config))) {
      signalPassFailure();
      return;
    }
    return;
  }
};
} // namespace

namespace xla_jit {
std::unique_ptr<mlir::Pass> createCleanFIROpsPass() {
  return std::make_unique<CleanFIROpsPass>();
}

void registerCleanFIROpsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createCleanFIROpsPass();
  });
};
} // namespace xla_jit
