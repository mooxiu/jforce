#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <cstdlib>
#include <regex>
#include <string>
#include <utility>
#include "Utils.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static bool isNotOutlineFunc(::mlir::StringRef funcName) {
  return !funcName.starts_with(JIT_OUTLINE_AFFINE_FUNC_PREFIX);
}

static std::string getCorrespondingStableHLOFuncName(::mlir::StringRef outlinedFuncName) {
  std::regex re(llvm::formatv("{0}[0-9]+$", JIT_OUTLINE_AFFINE_FUNC_PREFIX).str());
  if (!std::regex_match(outlinedFuncName.str(), re)) {
    llvm::errs() << "The outlinedFuncName does not match the pattern!\n";
    return "";
  }
  return llvm::formatv("{0}_raised", outlinedFuncName);
}

struct ReplaceOutlineFuncCall : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewritter) const final {
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    
    // llvm::DenseMap<func::FuncOp, func::FuncOp> outFuncPairs; 
    llvm::DenseMap<func::CallOp, ::mlir::StringRef> callToOutlineFuncMap;

    funcOp.walk([&](func::CallOp callOp){
      if(isNotOutlineFunc(callOp.getCallee())){
        // This call might be call some other built-in functions rather than built-in functions.
        return WalkResult::skip();
      }

      auto outlinedFuncName = callOp.getCallee();
      auto hloFuncName = getCorrespondingStableHLOFuncName(outlinedFuncName);
      if (hloFuncName.empty()) {
        return WalkResult::skip();
      }
      callToOutlineFuncMap[callOp] = outlinedFuncName;
      return WalkResult::advance();
    });

    for (const auto& entry: callToOutlineFuncMap) {
      auto callOp = entry.getFirst();     
      auto outlinedFuncName = entry.getSecond();

      func::FuncOp outlinedFunc = moduleOp.lookupSymbol<func::FuncOp>(outlinedFuncName);
      rewritter.setInsertionPoint(callOp);
      auto operands = callOp.getOperands();
      llvm::SmallVector<Value> tensorParams;
      for (const auto& operand: operands) {
        Value memrefParam = llvm::DynCastTo<MemRefType>(operand);
        bufferization::ToTensorOp::create(rewritter, callOp.getLoc(), TensorType(), operand);
      }
    };

    return failure();
  }
};


struct RemergePass: public mlir::PassWrapper<RemergePass, OperationPass<ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry & registry) const override {
  }

  StringRef getArgument() const override { 
    return "jforce-remerge";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto ctx = moduleOp.getContext();
    llvm::SmallVector<Operation*> maternalFunctions;
    moduleOp.walk([&](func::FuncOp funcOp){
      if (isNotOutlineFunc(funcOp.getName())) {
        maternalFunctions.push_back(funcOp);
      }
    });

    RewritePatternSet patterns(ctx);
    patterns.add<ReplaceOutlineFuncCall>(ctx);
    FrozenRewritePatternSet frozenSet(std::move(patterns));
    GreedyRewriteConfig config;
    config.isFoldingEnabled();

    if (failed(applyOpPatternsGreedily(maternalFunctions, frozenSet, config))) {
      signalPassFailure();
    }

    moduleOp.dump();
  }
};
};

namespace xla_jit {
std::unique_ptr<mlir::Pass> createRemergePass() {
  return std::make_unique<RemergePass>();
}

void registerRemergePass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRemergePass();
  });
};
} // namespace xla_jit
