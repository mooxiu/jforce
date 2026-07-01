#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "support/utilities.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

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

  LogicalResult matchAndRewrite(fir::ConvertOp op, PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    Value input = op.getOperand();
    Type fromTy = input.getType();
    Type toTy = op.getResult().getType();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);


    if (!(fromTy.isIntOrIndexOrFloat() && toTy.isIntOrIndexOrFloat()))
      return failure();

    if (fromTy == toTy) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (fromTy.isIndex()) {
      if (toTy.isIndex()) {
        rewriter.replaceOp(op, input);
        return success();
      }
      if (toTy.isInteger()) {
        rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, toTy, input);
        return success();
      }
      if (toTy.isFloat()) {
        Type i64Ty = rewriter.getI64Type();
        Value intVal = arith::IndexCastOp::create(rewriter, loc, i64Ty, input).getResult();
        rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, toTy, intVal);
        return success();
      }
    }

    if (fromTy.isInteger()) {
      if (toTy.isIndex()) {
        rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, toTy, input);
        return success();
      }
      if (toTy.isInteger()) {
        unsigned fromWidth = fromTy.getIntOrFloatBitWidth();
        unsigned toWidth = toTy.getIntOrFloatBitWidth();
        if (fromWidth < toWidth) {
          rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, toTy, input);
        } else if (fromWidth > toWidth) {
          rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, toTy, input);
        } else {
          rewriter.replaceOp(op, input);
        }
        return success();
      }

      if (toTy.isFloat()) {
        rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, toTy, input);
        return success();
      }
    }

    if (fromTy.isFloat()) {
      if (toTy.isFloat()) {
        unsigned fromWidth = fromTy.getIntOrFloatBitWidth();
        unsigned toWidth = toTy.getIntOrFloatBitWidth();
        if (fromWidth < toWidth) {
          rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, toTy, input);
        } else if (fromWidth > toWidth) {
          rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, toTy, input);
        } else {
          rewriter.replaceOp(op, input);
        }
        return success();
      }
      if (toTy.isInteger()) {
        rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, toTy, input);
        return success();
      }
      if (toTy.isIndex()) {
        Type i64Ty = rewriter.getI64Type();
        auto intVal = arith::FPToUIOp::create(rewriter, loc, i64Ty, input).getResult();
        rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, toTy, intVal);
        return success();
      }
    }

    DEBUG_PRINT("Unexpected convert op: ");
    DEBUG_PRINT_OP(op);
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
