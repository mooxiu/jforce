#include "Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

using namespace mlir;

/// The objective of this pass is to move affine for loop or other blocks in a
/// separate function, so the Enzyme-JAX's AffineToStableHLO pass can work.
namespace {

template <typename LoopType>
struct AffineLoopsToOutline : public OpRewritePattern<LoopType> {
  using OpRewritePattern<LoopType>::OpRewritePattern;

  static llvm::SetVector<Value> getPredefinedVals(LoopType loopOp) {
    // Collect all values defined outside of the affineOp itself.
    llvm::SetVector<Value> preloopDefinedVals;

    // Values defined outside of forOp used in forOp region.
    mlir::getUsedValuesDefinedAbove({loopOp.getRegion()}, preloopDefinedVals);
    // Insert values of forOp itself.
    llvm::for_each(loopOp.getOperands(),
                   [&](Value forOpVal) { preloopDefinedVals.insert(forOpVal); });
    return preloopDefinedVals;
  }

  static func::FuncOp createOutlineFunc(
    func::FuncOp funcOp, 
    LoopType loopOp, 
    PatternRewriter &rewritter,
    const llvm::SmallVector<Type> &outlinedFuncInputTypes,
    const llvm::SmallVector<Type> &outlinedFuncOutputTypes
  ) {
    rewritter.setInsertionPoint(funcOp);
    auto outlinedFuncType = rewritter.getFunctionType(outlinedFuncInputTypes, outlinedFuncOutputTypes);
    auto outlinedFuncName = llvm::formatv("outlined_affinefor_{0}",
                                          reinterpret_cast<std::uintptr_t>(loopOp.getAsOpaquePointer())).str();
    llvm::SmallVector<NamedAttribute> attrs = {};
    llvm::SmallVector<DictionaryAttr> argAttrs = {};
    auto outlinedFunc = func::FuncOp::create(rewritter, funcOp.getLoc(), outlinedFuncName, outlinedFuncType, attrs, argAttrs);
    return outlinedFunc;
  }

  static void prepareCallOp(
    MLIRContext *ctx, LoopType loopOp,
    PatternRewriter &rewritter,
    llvm::SmallVector<Value> &realInputArgs,
    llvm::DenseMap<Value, Value> &affineResToMem,
    const llvm::SetVector<Value> &preloopDefinedVals
  ) {
    rewritter.setInsertionPoint(loopOp);
    for (auto val: preloopDefinedVals) {
      if (matchPattern(val, m_Constant())) {
        // DO NOTHING (constants are supposed to be already folded, so they will
        // not be used as arguments of outlined function). (Compute args will
        // not be folded in the outer function as constants, they will be
        // wrapped by memref usually).
      } else if (llvm::isa<mlir::MemRefType>(val.getType())) {
        realInputArgs.push_back(val);
      } else if (val.getType().isIndex()) {
        Type elementType = IntegerType::get(ctx, 64);
        auto castOp = arith::IndexCastOp::create(rewritter, loopOp.getLoc(), elementType, val);
        auto allocaOp = memref::AllocaOp::create(rewritter, loopOp.getLoc(), MemRefType::get({}, elementType, {}, {}));
        auto storeOp = memref::StoreOp::create(rewritter, loopOp.getLoc(), castOp.getResult(), allocaOp.getResult(), {});
        realInputArgs.push_back(storeOp.getMemRef());
      } else if (val.getType().isIntOrFloat()) {
        auto allocaOp = memref::AllocaOp::create(rewritter, loopOp.getLoc(), MemRefType::get({}, val.getType(), {}, {}));
        auto storeOp = memref::StoreOp::create(rewritter, loopOp.getLoc(), val, allocaOp.getResult(), {});
        realInputArgs.push_back(storeOp.getMemRef());
      } else {
        llvm::errs() << "Unexpected Type!\n";
        val.dump();
        std::exit(EXIT_FAILURE);
      }
    }
    for (auto val: loopOp.getResults()) {
      if (val.getType().isIntOrFloat()) {
        auto allocaOp = memref::AllocaOp::create(rewritter, loopOp.getLoc(), MemRefType::get({}, val.getType(), {}, {}));
        realInputArgs.push_back(allocaOp.getMemref());
        affineResToMem.insert(std::pair(val, allocaOp.getMemref()));
      } else {
        llvm::errs() << "Unexpected Type!\n";
        val.dump();
        std::exit(EXIT_FAILURE);
      }
    }
  }

  static void fillOutlinedFunc(MLIRContext *ctx, LoopType loopOp,
                               PatternRewriter &rewritter, Block *entryBlock,
                               const llvm::SetVector<Value> &preloopDefinedVals,
                               const llvm::SmallVector<Value> &realInputArgs) {
    rewritter.setInsertionPointToEnd(entryBlock);
    IRMapping mapping;
    int outlinedFuncIdx = 0;
    for (int i = 0; i < preloopDefinedVals.size(); i++) {
      auto outVal = preloopDefinedVals[i];
      auto blockArg = entryBlock->getArgument(outlinedFuncIdx);
      outlinedFuncIdx += 1;

      mlir::IntegerAttr attr;
      if (matchPattern(outVal, m_Constant(&attr))) {
        outlinedFuncIdx -= 1;
        if (outVal.getType().isIndex()) {
          auto constIndexOp = arith::ConstantIndexOp::create(
              rewritter, loopOp.getLoc(), attr.getInt());
          mapping.map(outVal, constIndexOp.getResult());
        } else if (outVal.getType().isInteger()) {
          auto constIntOp = arith::ConstantIntOp::create(
              rewritter, loopOp.getLoc(), attr.getInt(),
              outVal.getType().getIntOrFloatBitWidth());
          mapping.map(outVal, constIntOp.getResult());
        } else {
          llvm::errs() << "Unexpected value type!\n";
          std::exit(EXIT_FAILURE);
        }
      } else if (llvm::isa<mlir::MemRefType>(outVal.getType())) {
        mapping.map(outVal, blockArg);
      } else if (outVal.getType().isIndex()) {
        // auto loadOp = memref::LoadOp::create(rewritter, forOp.getLoc(),
        // blockArg, {});
        auto loadOp = affine::AffineLoadOp::create(
            rewritter, loopOp.getLoc(), AffineMap::get(ctx), blockArg);
        auto castBackOp =
            arith::IndexCastOp::create(rewritter, loopOp.getLoc(),
                                       IndexType::get(ctx), loadOp.getResult());
        mapping.map(outVal, castBackOp.getResult());
      } else if (outVal.getType().isIntOrFloat()) {
        // auto loadOp = memref::LoadOp::create(rewritter, forOp.getLoc(),
        // blockArg, {});
        auto loadOp = affine::AffineLoadOp::create(
            rewritter, loopOp.getLoc(), AffineMap::get(ctx), blockArg);
        mapping.map(outVal, loadOp.getResult());
      } else {
        llvm::errs() << "Should not go here.\n";
        std::exit(EXIT_FAILURE);
      }
    }

    for (int i = 0; i < realInputArgs.size(); i++) {
      mapping.map(*(realInputArgs.begin() + i), entryBlock->getArgument(i));
    }
    auto cloned = rewritter.clone(*loopOp.getOperation(), mapping);
    for (const auto& val: cloned->getResults()) {
      auto blockArg = entryBlock->getArgument(outlinedFuncIdx);
      affine::AffineStoreOp::create(rewritter, loopOp.getLoc(), val, blockArg, {});
      outlinedFuncIdx += 1;
    }
    func::ReturnOp::create(rewritter, loopOp.getLoc()); // the returnOp should be empty
  }

  static Type convertToParamType(MLIRContext* ctx, Value val) {
    auto valType = val.getType();
    auto valTypeInfo = inspectTypeInfo(valType);
    if (matchPattern(val, m_Constant())) {
      return nullptr;
    } else if (llvm::isa<mlir::MemRefType>(valType)) {
      return valType;
    } else if (valType.isIntOrFloat()) {
      return MemRefType::get(valTypeInfo.shape, valTypeInfo.elementTy, {}, {});
    } else if (valType.isIndex()) {
      return MemRefType::get(valTypeInfo.shape, IntegerType::get(ctx, 64), {}, {});
    } else {
      llvm::errs() << "Cannot handle this!\n";
      val.dump();
      std::exit(EXIT_FAILURE);
    }
  }

  LogicalResult matchAndRewrite(LoopType loopOp,
                                PatternRewriter &rewritter) const final {
    if (loopOp -> template getParentOfType<affine::AffineForOp>() 
        || loopOp -> template getParentOfType<affine::AffineParallelOp>()) {
      return failure();
    }

    func::FuncOp funcOp = loopOp ->template getParentOfType<func::FuncOp>();
    if (!funcOp) {
      llvm::errs() << "loopOp should be wrapped in a function Op!\n";
      std::exit(EXIT_FAILURE);
    }
    auto ctx = rewritter.getContext();

    // Including defined values before the loop, loop operands.
    llvm::SetVector<Value> dependVals;
    mlir::getUsedValuesDefinedAbove({loopOp.getRegion()}, dependVals);
    llvm::for_each(loopOp.getOperands(), [&](Value forOpVal) { dependVals.insert(forOpVal); });
    auto loopReturnedVals  = loopOp.getResults();

    llvm::SmallVector<Type> outlinedFuncInputTypes;
    llvm::for_each(dependVals, [&](Value val){
      auto ty = convertToParamType(ctx, val);
      if (ty) {
        outlinedFuncInputTypes.push_back(ty);
      }
    });
    llvm::for_each(loopReturnedVals, [&](Value val){
      auto ty = convertToParamType(ctx, val);
      if (ty) {
        outlinedFuncInputTypes.push_back(ty);
      }
    });
    auto outlinedFunc = createOutlineFunc(funcOp, loopOp, rewritter, outlinedFuncInputTypes, {});
    auto entryBlock = outlinedFunc.addEntryBlock();

    llvm::SmallVector<Value> realInputArgs;
    llvm::DenseMap<Value, Value> affineResToMem;
    prepareCallOp(ctx, loopOp, rewritter, realInputArgs, affineResToMem, dependVals);
    fillOutlinedFunc(ctx, loopOp, rewritter, entryBlock, dependVals, realInputArgs);

    rewritter.setInsertionPoint(loopOp);
    func::CallOp::create(rewritter, loopOp.getLoc(), outlinedFunc, realInputArgs);
    rewritter.setInsertionPointAfter(loopOp);
    for (const auto& entry: affineResToMem) {
      auto loadOp = affine::AffineLoadOp::create(rewritter, loopOp.getLoc(), entry.getSecond(), {});
      rewritter.replaceAllUsesWith(entry.getFirst(), loadOp.getResult());
    }

    rewritter.eraseOp(loopOp);
    return success();
  }
};

struct OutlineAffinePass
    : public mlir::PassWrapper<OutlineAffinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-outline-affine"; }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *ctx = moduleOp->getContext();
    OpBuilder opBuilder(ctx);

    llvm::SmallVector<func::FuncOp> funcOps;
    moduleOp.walk([&](func::FuncOp funcOp) { funcOps.push_back(funcOp); });

    RewritePatternSet patterns(ctx);
    patterns.add<AffineLoopsToOutline<affine::AffineForOp>>(ctx);
    patterns.add<AffineLoopsToOutline<affine::AffineParallelOp>>(ctx);
    GreedyRewriteConfig config;
    config.enableFolding();

    FrozenRewritePatternSet frozenRewritePatternSet(std::move(patterns));

    llvm::for_each(funcOps, [&](func::FuncOp funcOp) {
      if (failed(
              applyPatternsGreedily(funcOp, frozenRewritePatternSet, config))) {
        signalPassFailure();
        return;
      }
    });
    moduleOp.dump();
    return;
  }
};
} // namespace

namespace xla_jit {
std::unique_ptr<mlir::Pass> createOutlineAffinePass() {
  return std::make_unique<OutlineAffinePass>();
}

void registerOutlineAffinePass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createOutlineAffinePass();
  });
};
} // namespace xla_jit
