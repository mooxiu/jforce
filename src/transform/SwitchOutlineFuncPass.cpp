#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include <cassert>
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
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
    llvm::dbgs() << "The outlinedFuncName is: " << outlinedFuncName << ", which does not match the pattern. Skip this callee.\n";
    return "";
  }
  return llvm::formatv("{0}_raised", outlinedFuncName);
}

struct ReplaceOutlineFuncCall : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  static void replaceMemrefFuncCall(
    ModuleOp moduleOp,
    PatternRewriter &rewritter,
    func::CallOp callOp, 
    ::mlir::StringRef outlinedFuncName
  ) {
    func::FuncOp outlinedFunc = moduleOp.lookupSymbol<func::FuncOp>(outlinedFuncName);
    auto hloFuncName = llvm::formatv("{0}_raised", outlinedFuncName).str(); 
    func::FuncOp hloFunc = moduleOp.lookupSymbol<func::FuncOp>(hloFuncName);
    assert(outlinedFunc.getNumArguments() == hloFunc.getNumArguments());
    assert(outlinedFunc.getNumResults() == 0);
    assert(hloFunc.getNumArguments() == hloFunc.getNumResults());

    rewritter.setInsertionPoint(callOp);
    auto memrefFuncParams = callOp.getOperands();
    int paramsSize = memrefFuncParams.size(); 
    assert(paramsSize == outlinedFunc.getNumArguments());

    llvm::SmallVector<Value> tensorParams;
    for (int paramIdx = 0; paramIdx < paramsSize; paramIdx++) {
      auto memrefFuncParam = memrefFuncParams[paramIdx];
      MemRefType operandMemrefType = llvm::dyn_cast<MemRefType>(memrefFuncParam.getType());
      auto operandTensorType = RankedTensorType::get(operandMemrefType.getShape(), operandMemrefType.getElementType());
      auto memrefTypeToTensorTypeOp = bufferization::ToTensorOp::create(rewritter, callOp.getLoc(), operandTensorType, memrefFuncParam);
      tensorParams.push_back(memrefTypeToTensorTypeOp);
    }
    auto hloFuncCallOp = func::CallOp::create(rewritter, callOp.getLoc(), hloFunc, tensorParams);
    
    rewritter.setInsertionPointAfter(callOp);
    for (int paramIdx = 0; paramIdx < paramsSize; paramIdx++) {
      auto hloFuncCallOpRes = hloFuncCallOp.getResult(paramIdx);
      // memref.tensor_store has been replaced with bufferization.materialize_in_destination, see: https://github.com/llvm/llvm-project/pull/71010
      if (llvm::isa<MemRefType>(memrefFuncParams[paramIdx].getType())) {
        bufferization::MaterializeInDestinationOp::create(
            rewritter,
            callOp.getLoc(), 
            mlir::TypeRange{},
            hloFuncCallOpRes,
            memrefFuncParams[paramIdx],
            false,
            true
        );
      }
    }
    
    rewritter.eraseOp(callOp);
    rewritter.eraseOp(outlinedFunc);
    return;
  }


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


    if (callToOutlineFuncMap.empty()) {
      return failure();
    }

    for (const auto& entry: callToOutlineFuncMap) {
      auto callOp = entry.getFirst();     
      auto outlinedFuncName = entry.getSecond();
      replaceMemrefFuncCall(moduleOp, rewritter, callOp, outlinedFuncName);
    };

    llvm::dbgs() << "\nSwitchOutlineFuncPass: swicth one!\n";
    return success();
  }
};


struct SwitchOutlineFuncPass: public mlir::PassWrapper<SwitchOutlineFuncPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwitchOutlineFuncPass)

  void getDependentDialects(mlir::DialectRegistry & registry) const override {
    registry.insert<
        bufferization::BufferizationDialect,
        memref::MemRefDialect
      >();
  }

  StringRef getArgument() const override { 
    return "jforce-switch-outline-function";
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
    config.enableFolding();

    if (failed(applyOpPatternsGreedily(maternalFunctions, frozenSet, config))) {
      signalPassFailure();
    }
    return;
  }
};
};

namespace xla_jit {
std::unique_ptr<mlir::Pass> createRemergePass() {
  return std::make_unique<SwitchOutlineFuncPass>();
}

void registerRemergePass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRemergePass();
  });
};
} // namespace xla_jit
