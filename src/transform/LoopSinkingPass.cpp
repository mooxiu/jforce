/// Objective of this pass is to construct pefect nested loop for latter affine
/// operations.
///
/// Before Example:
///   affine.for %arg14 = 1 to 4 {
///     %19 = arith.index_cast %arg14 : index to i32
///     affine.for %arg15 = 1 to 5 {
///       %20 = arith.index_cast %arg15 : index to i32
///       %21 = arith.index_cast %20 : i32 to index
///       %22 = arith.subi %21, %c1 : index
///       %23 = arith.index_cast %19 : i32 to index
///       %24 = arith.subi %23, %c1 : index
///       %25 = memref.load %13[%24, %22] : memref<3x4xf64>
///       %26 = arith.mulf %16, %25 fastmath<contract> : f64
///       %27 = memref.load %14[%24, %22] : memref<3x4xf64>
///       %28 = arith.addf %26, %27 fastmath<contract> : f64
///       memref.store %28, %15[%24, %22] : memref<3x4xf64>
///     }
///   }
///
/// After Example:
///   affine.for %arg14 = 1 to 4 {
///     affine.for %arg15 = 1 to 5 {
///       %19 = arith.index_cast %arg14 : index to i32
///       %20 = arith.index_cast %arg15 : index to i32
///       %21 = arith.index_cast %20 : i32 to index
///       %22 = arith.subi %21, %c1 : index
///       %23 = arith.index_cast %19 : i32 to index
///       %24 = arith.subi %23, %c1 : index
///       %25 = memref.load %13[%24, %22] : memref<3x4xf64>
///       %26 = arith.mulf %16, %25 fastmath<contract> : f64
///       %27 = memref.load %14[%24, %22] : memref<3x4xf64>
///       %28 = arith.addf %26, %27 fastmath<contract> : f64
///       memref.store %28, %15[%24, %22] : memref<3x4xf64>
///     }
///   }

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace mlir;
namespace {

// TODO: change this to also include affine::AffineParallelOp using template!
// Before:
//  loop1 {
//    ...statements...
//    loop2 {
//    }
//  }
//
// After:
//  loop1 {
//    loop2 {
//      ...statements...
//    }
//  }
//
// Precondition:
// - Statements has no side effect (like reading or storing to mem)
// - Maybe I should also assert both loops are affine so this pass is
// meaningful?
struct SinkToInnerLoop : OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp innerLoop,
                                PatternRewriter &rewriter) const final {
    auto outerLoop = innerLoop->getParentOfType<affine::AffineForOp>();
    if (!outerLoop) {
      return failure();
    }

    llvm::SmallVector<Operation *> operationsToMove;
    auto currOp = innerLoop->getPrevNode();
    while (currOp) {
      if (!isMemoryEffectFree(currOp)) {
        return failure();
      }
      operationsToMove.push_back(currOp);
      currOp = currOp->getPrevNode();
    }
    if (operationsToMove.empty()) {
      return failure();
    }

    for (auto opToMove: operationsToMove) {
      rewriter.moveOpBefore(opToMove, innerLoop.getBody(), innerLoop.getBody()->begin());
    }
    return success();
  }
};

struct LoopSinkingPass
    : public mlir::PassWrapper<LoopSinkingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopSinkingPass)

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
    patterns.add<SinkToInnerLoop>(ctx);
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
} // namespace

namespace xla_jit {
std::unique_ptr<mlir::Pass> createLoopSinkingPass() {
  return std::make_unique<LoopSinkingPass>();
}

void registerLoopSinkingPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createLoopSinkingPass(); });
};
} // namespace xla_jit
