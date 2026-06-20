/// INFO: the objective of this is to conclude the common part of `OptimizingMemOps` and `IfConversion`. 

// FIX: we analysis the alias of 2 mem, but can the mem be array? we probably get false positive. For example, A[i] and A[j] have same memref, but A[i] and A[j] are different.
// FIX: other false positive include: a function call with arguments which cannot be alias.
// FIX: other false positive include: a control flow does not contain any alias.

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "transform/Utils.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstdlib>

using namespace mlir;

namespace {

// Mem2reg is supposed to cover alloca, but we need aliasing analysis here.
//
// Before:
// fir.store %8 to %0#0 : !fir.ref<i32>
// ...other operations
// %10 = fir.load %0#0 : !fir.ref<i32>
// .. usages of %10
//
// After:
// fir.store %8 to %0#0 : !fir.ref<i32>
// .. usages of %8
//
// Precondition:
// - store and read in same block: avoid complicated control flow analysis
// - there is no control flow or function calls between the 2 operations
// - there is no assignment operations between the 2 operations that assign to the aliased mem
//
// Why correct:
// store a number to an address and load it can be forwarded if nothing changes.
//
// Notice that we do not erase the store operation as it can be DCE-ed if it is
// not been used.

// template <typename LoadTy, typename StoreTy>
struct ReduceWriteAndReadSameAddr : public OpRewritePattern<fir::LoadOp> {
private:
  fir::AliasAnalysis& aliasAnalysis;

public:
  ReduceWriteAndReadSameAddr(mlir::MLIRContext* ctx, fir::AliasAnalysis& aa)
    : OpRewritePattern<fir::LoadOp>(ctx), aliasAnalysis(aa) {}

  // using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(fir::LoadOp loadOp, PatternRewriter &rewriter) const final {
    while (Operation* prev = loadOp->getPrevNode()) {
      if (auto storeOp = llvm::dyn_cast<fir::StoreOp>(prev)) {
        auto res = aliasAnalysis.alias(storeOp.getMemref(), loadOp.getMemref());
        if (res.isMust()) {
          rewriter.replaceAllUsesWith(loadOp.getResult(), storeOp.getValue());
          return success();
        }
      }
      if (mayWriteToMemory(loadOp.getMemref(), prev, aliasAnalysis)) {
        return failure();
      }
      // control flow, function call, 
      if (llvm::isa<func::CallOp>(prev)) {
        return failure();
      } 
      if (prev->getNumRegions() > 0) {
        return failure();
      }
    }
    return failure();
  }
};


// Before:
//   %13 = fir.load %8[] : memref<i32>
//   fir.store %13, %8[] : memref<i32>
// After:
//    (only delete the store, and hope the load will be erased in other pattern matchings)
//   %13 = memref.load %8[] : memref<i32>
//
// Precondition:
//   - address should be non aliasing
//   - between the load and store, there should not be any other store to the addr!
// TODO: rename to `FoldStoreToReadMem`
struct ReduceReadAndWriteSameAddr : public OpRewritePattern<fir::StoreOp> {
private:
  fir::AliasAnalysis& aliasAnalysis;

public:
  ReduceReadAndWriteSameAddr(mlir::MLIRContext* ctx, fir::AliasAnalysis& aa)
    : OpRewritePattern<fir::StoreOp>(ctx), aliasAnalysis(aa) {}

  // using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(fir::StoreOp storeOp, PatternRewriter &rewriter) const final {
    while(auto prev = storeOp->getPrevNode()) {
      if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(prev)) {
        auto res = aliasAnalysis.alias(loadOp.getMemref(), storeOp.getMemref());
        if (res.isMust()) {
          rewriter.eraseOp(storeOp);
          return success();
        }
      }

      // "We have load the value from the memory, and have never changed the memory"
      if (mayWriteToMemory(storeOp.getMemref(), prev, aliasAnalysis)) {
        return failure();
      } 
      // control flow, function call, 
      if (llvm::isa<func::CallOp>(prev)) {
        return failure();
      } 
      if (prev->getNumRegions() > 0) {
        return failure();
      }
    }
    return failure();
  }
};
 

// Fold repeated storeOp to the same memory location. 
//
// Before:
//  fir.store %8 to %0#0
//  ...
//  fir.store %18 to %0#0
//
// After:
//  ...
//  fir.store %18 to %0#0
//  ...
//
// Precondition:
// - both store operations in the same block.(simplify analysis)
// - two store ops save values to same mems or must alias mems
// - there should be no other usage (read or write) of the mem which may alias the mem
// - no call function and control flows in between (save escape analysis)
struct FoldRepeatStoreOps: public OpRewritePattern<fir::StoreOp> {
private:
  fir::AliasAnalysis& aliasAnalysis;

public:
  FoldRepeatStoreOps(mlir::MLIRContext* ctx, fir::AliasAnalysis& aa)
    : OpRewritePattern<fir::StoreOp>(ctx), aliasAnalysis(aa) {}

  // using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(fir::StoreOp storeOp, PatternRewriter &rewriter) const final {
    while(auto prev = storeOp->getPrevNode()) {
      if (auto repeatStoreOp = llvm::dyn_cast<fir::StoreOp>(prev)) {
        if (aliasAnalysis.alias(storeOp.getMemref(), repeatStoreOp.getMemref()).isMust()) {
          rewriter.eraseOp(repeatStoreOp);
          return success();
        }
      }
      if (mayAccessMemory(storeOp.getMemref(), prev, aliasAnalysis)) {
        return failure();
      }
      if (llvm::isa<func::CallOp>(prev)) {
        return failure();
      } 
      if (prev->getNumRegions() > 0) {
        return failure();
      }
    }
    return failure();
  }
};



struct MemOpsFoldingPass
    : public mlir::PassWrapper<MemOpsFoldingPass,
                               mlir::OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-mem-ops-folding"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = getOperation()->getContext();

    bool hasHLFIR = false;
    funcOp.walk([&](mlir::Operation *op) {
      if (op->getDialect()->getNamespace() == "hlfir") {
        hasHLFIR = true;
        return mlir::WalkResult::interrupt(); 
      }
      return mlir::WalkResult::advance();
    });

    if (hasHLFIR) {
      funcOp.emitError("Precondition failed: HLFIR operations are not allowed in this pass!");
      return signalPassFailure();
    }

    fir::AliasAnalysis aliasAnalysis;
    RewritePatternSet patterns(ctx);
    patterns.add<ReduceWriteAndReadSameAddr>(ctx, aliasAnalysis);
    patterns.add<ReduceReadAndWriteSameAddr>(ctx, aliasAnalysis);
    patterns.add<FoldRepeatStoreOps>(ctx, aliasAnalysis);
    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
  }
};
}; // namespace

namespace xla_jit {
std::unique_ptr<mlir::Pass> createMemOpsFoldingPass() {
  return std::make_unique<MemOpsFoldingPass>();
}

void registerMemOpsFoldingPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createMemOpsFoldingPass(); });
}
} // namespace xla_jit
