// TODO: maybe should do this in compile time

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>

using namespace mlir;

struct RedeclaredVars:  public OpRewritePattern<hlfir::DeclareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hlfir::DeclareOp declareOp,
                                PatternRewriter &rewriter) const final {
    auto memref = declareOp.getMemref();
    if (!memref.getDefiningOp()) {
      return failure();
    }
    if (auto originDefineOp = llvm::dyn_cast<hlfir::DeclareOp>(memref.getDefiningOp())) {
      assert(originDefineOp.getNumResults() == 2);
      assert(declareOp.getNumResults() == 2);
      declareOp.getResult(0).replaceAllUsesWith(originDefineOp.getResult(0));
      declareOp.getResult(1).replaceAllUsesWith(originDefineOp.getResult(1));
      rewriter.eraseOp(declareOp);
      return success();
    }
    return failure();
  }
};

struct FoldAssignToTempVarsInLoop : public OpRewritePattern<hlfir::DeclareOp> {
   using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hlfir::DeclareOp declareOp,
                                PatternRewriter &rewriter) const final {
    auto memref = declareOp.getMemref();
    if (!memref.getDefiningOp()) {
      return failure();
    }
    if (llvm::isa<fir::AllocaOp>(memref.getDefiningOp())) {
      // this is a temp
      llvm::SmallVector<Operation*> usersInsideLoop;
      for(auto user : declareOp.getResults().getUsers()) {
        if (user->getParentOfType<fir::DoLoopOp>()) {
          usersInsideLoop.push_back(user);
        }
      }
      
      bool hasChange = false;
      if (usersInsideLoop.size() == 1) {
        Operation* op = usersInsideLoop[0];
        if (llvm::isa<hlfir::AssignOp>(op)) {
          rewriter.eraseOp(op);
          hasChange = true;
        }
      } 
      if (declareOp.getResults().getUses().empty()) {
        rewriter.eraseOp(declareOp);
        hasChange = true;
        if (memref.getDefiningOp()->getResults().getUses().empty()) {
          rewriter.eraseOp(memref.getDefiningOp());
        }
      }
      if (hasChange) return success();
    }
    return failure();
  }
};

// struct FoldTempVars: public OpRewritePattern<>

namespace {
struct CleanTempsPass
    : public PassWrapper<CleanTempsPass, mlir::OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CleanTempsPass)

  StringRef getArgument() const override { return "jforce-clean-temps"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RedeclaredVars>(ctx);
    patterns.add<FoldAssignToTempVarsInLoop>(ctx);
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
std::unique_ptr<mlir::Pass> createCleanTempsPass() {
  return std::make_unique<CleanTempsPass>();
}

void registerCleanTempsPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createCleanTempsPass(); });
};
} // namespace xla_jit
