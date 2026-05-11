#include "Utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>

using namespace mlir;

/// The objective of this pass is to move affine for loop or other blocks in a
/// separate function, so the Enzyme-JAX's AffineToStableHLO pass can work.
namespace {

// fir.convert index -> i32/64 should be arith.index_case
// fir.i32 -> i64 should be arith.extsi or arith.extui
// fir.i64 -> i32 should be arith.trunci
struct ReplaceFIRConvert : public OpRewritePattern<fir::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(fir::ConvertOp convertOp,
                                PatternRewriter &rewriter) const final {
    auto fromVal = convertOp.getOperand();
    auto fromValType = fromVal.getType();
    auto toVal = convertOp.getResult();
    auto toValType = toVal.getType();

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

static func::FuncOp outlineAffineForOp(MLIRContext *ctx, func::FuncOp funcOp,
                                       OpBuilder &opBuilder,
                                       affine::AffineForOp forOp) {
  // Collect all values defined outside of the affineOp itself.
  llvm::SetVector<Value> outDefinedVals;
  // Values defined outside of forOp used in forOp region.
  mlir::getUsedValuesDefinedAbove({forOp.getRegion()}, outDefinedVals);
  // Insert values of forOp itself.
  llvm::for_each(forOp.getOperands(),
                 [&](Value forOpVal) { outDefinedVals.insert(forOpVal); });

  int inputArgSize = outDefinedVals.size();
  llvm::SmallVector<Type> outlinedFuncInputTypes(inputArgSize);
  for (int i = 0; i < inputArgSize; i++) {
    auto valType = outDefinedVals[i].getType();
    auto valTypeInfo = inspectTypeInfo(valType);
    if (llvm::isa<mlir::MemRefType>(valType)) {
      outlinedFuncInputTypes[i] = valType;
    } else if (valType.isIntOrFloat()) {
      outlinedFuncInputTypes[i] =
          MemRefType::get(valTypeInfo.shape, valTypeInfo.elementTy, {}, {});
    } else if (valType.isIndex()) {
      outlinedFuncInputTypes[i] =
          MemRefType::get(valTypeInfo.shape, IntegerType::get(ctx, 32), {}, {});
    } else {
      llvm::errs() << "Cannot handle this!\n";
      std::exit(EXIT_FAILURE);
    }
  }
  opBuilder.setInsertionPoint(funcOp);
  auto outlinedFuncType = opBuilder.getFunctionType(outlinedFuncInputTypes, {});
  auto outlinedFuncName =
      llvm::formatv("outlined_affinefor_{0}", reinterpret_cast<std::uintptr_t>(
                                                  forOp.getAsOpaquePointer()))
          .str();
  llvm::SmallVector<NamedAttribute> attrs = {};
  llvm::SmallVector<DictionaryAttr> argAttrs = {};
  auto outlinedFunc =
      func::FuncOp::create(opBuilder, funcOp.getLoc(), outlinedFuncName,
                           outlinedFuncType, attrs, argAttrs);
  auto entryBlock = outlinedFunc.addEntryBlock();
  opBuilder.setInsertionPointToEnd(entryBlock);

  // Copy from old to new
  opBuilder.setInsertionPoint(forOp);
  llvm::SmallVector<Value> realInputArgs(inputArgSize);
  for (int i = 0; i < inputArgSize; i++) {
    auto outVal = outDefinedVals[i];
    if (llvm::isa<mlir::MemRefType>(outVal.getType())) {
      realInputArgs[i] = outVal;
    } else if (outVal.getType().isIndex()) {
      Type elementType = IntegerType::get(ctx, 32);
      auto castOp = arith::IndexCastOp::create(opBuilder, forOp.getLoc(),
                                               elementType, outVal);
      auto allocaOp = memref::AllocaOp::create(
          opBuilder, forOp.getLoc(), MemRefType::get({}, elementType, {}, {}));
      auto storeOp =
          memref::StoreOp::create(opBuilder, forOp.getLoc(), castOp.getResult(),
                                  allocaOp.getResult(), {});
      realInputArgs[i] = storeOp.getMemRef();
    } else if (outVal.getType().isIntOrFloat()) {
      auto allocaOp = memref::AllocaOp::create(
          opBuilder, forOp.getLoc(),
          MemRefType::get({}, outVal.getType(), {}, {}));
      auto storeOp = memref::StoreOp::create(opBuilder, forOp.getLoc(), outVal,
                                             allocaOp.getResult(), {});
      realInputArgs[i] = storeOp.getMemRef();
    } else {
      llvm::errs() << "Unexpected Type!\n";
      std::exit(EXIT_FAILURE);
    }
  }
  // add Call function.
  func::CallOp::create(opBuilder, forOp.getLoc(), outlinedFunc, realInputArgs);

  // Insert to the outlined function.
  opBuilder.setInsertionPointToEnd(entryBlock);
  IRMapping mapping;
  for (int i = 0; i < inputArgSize; i++) {
    auto outVal = outDefinedVals[i];
    auto blockArg = entryBlock->getArgument(i);
    if (llvm::isa<mlir::MemRefType>(outVal.getType())) {
      mapping.map(outVal, blockArg);
    } else if (outVal.getType().isIndex()) {
      // auto loadOp = memref::LoadOp::create(opBuilder, forOp.getLoc(),
      // blockArg, {});
      auto loadOp = affine::AffineLoadOp::create(opBuilder, forOp.getLoc(),
                                                 AffineMap::get(ctx), blockArg);
      auto castBackOp = arith::IndexCastOp::create(
          opBuilder, forOp.getLoc(), IndexType::get(forOp.getContext()),
          loadOp.getResult());
      mapping.map(outVal, castBackOp.getResult());
    } else if (outVal.getType().isIntOrFloat()) {
      // auto loadOp = memref::LoadOp::create(opBuilder, forOp.getLoc(),
      // blockArg, {});
      auto loadOp = affine::AffineLoadOp::create(opBuilder, forOp.getLoc(),
                                                 AffineMap::get(ctx), blockArg);
      mapping.map(outVal, loadOp.getResult());
    } else {
      llvm::errs() << "Should not go here.\n";
      std::exit(EXIT_FAILURE);
    }
  }

  for (int i = 0; i < inputArgSize; i++) {
    mapping.map(*(realInputArgs.begin() + i), entryBlock->getArgument(i));
  }
  opBuilder.clone(*forOp.getOperation(), mapping);
  func::ReturnOp::create(opBuilder,
                         outlinedFunc.getLoc()); // the returnOp should be empty
  return outlinedFunc;
}

struct OutlineAffinePass
    : public mlir::PassWrapper<OutlineAffinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-outline-affine"; }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext* ctx = getOperation()->getContext();
    OpBuilder opBuilder(moduleOp.getContext());

    llvm::SmallVector<func::FuncOp> funcOps;
    moduleOp.walk([&](func::FuncOp funcOp) { funcOps.push_back(funcOp); });

    llvm::SmallVector<func::FuncOp> outlinedFuncs;
    for (auto funcOp : funcOps) {
      llvm::SmallVector<Operation *> toDelete;
      funcOp.walk([&](affine::AffineForOp affineForOp) {
        // creating a function, which has the inputs for all the slices and affine bounds.
        outlinedFuncs.push_back(outlineAffineForOp(ctx, funcOp, opBuilder, affineForOp));
        toDelete.push_back(affineForOp);
        return;
      });
      for (auto *op : toDelete) {
        op->erase();
      }
    }

    RewritePatternSet patterns(ctx);
    patterns.add<ReplaceFIRConvert>(ctx);
    GreedyRewriteConfig config;
    config.enableFolding();
    llvm::for_each(outlinedFuncs, [&](auto outlinedFunc) {
      if (failed(applyPatternsGreedily(outlinedFunc, std::move(patterns), config))) {
        signalPassFailure();
      }
    });
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
