#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace mlir;

namespace {
// Before:
// fir.store %8 to %0#0 : !fir.ref<i32>
// %10 = fir.load %0#0 : !fir.ref<i32>
// .. usages of %10
// After:
// fir.store %8 to %0#0 : !fir.ref<i32>
// .. usages of %8
//
// Precondition:
// - no aliasing of %0#0 (assuming)
// - %0#0 is private (assuming)
// - no other fir.store to %0#0
// - no function call use %0#0 as parameter
//
// Why correct:
// store a number to an address and load it can be forwarded if nothing changes. 
//
struct ReduceLoadStoredAddress : public OpRewritePattern<fir::LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(fir::LoadOp loadOp,
                                PatternRewriter &rewriter) const final {
    auto addr = loadOp.getMemref();
    Operation* currOp = loadOp;
    while ((currOp = currOp->getPrevNode())) {
      if (auto storeOp = llvm::dyn_cast<fir::StoreOp>(currOp)) {
        if (storeOp.getMemref() == addr) {
          rewriter.replaceOp(loadOp, storeOp.getValue());
          return success();
        } else {
          continue;
        }
      }

      if (auto callOp = llvm::dyn_cast<func::CallOp>(currOp)) {
        bool addrUsedInFuncCall = false;
        llvm::for_each(callOp.getOperands(), [&](auto parameter){
          // TODO: need memory aliasing analysis if want to be more strict
          if (parameter == addr) {
            addrUsedInFuncCall = true;
          }
          return;
        });
        if (addrUsedInFuncCall) {
          return failure();
        } else {
          continue;
        }
      }
    }
    return failure();
  };
};

// Before:
// fir.do_loop .. {
//  ...
//  fir.store %8 to %0#0
//  ...
//  fir.store %18 to %0#0
//  ...
// }
// 
// After:
// fir.do_loop .. {
//  ...
//  fir.store %18 to %0#0
//  ...
// }
//
// Precondition:
// - Between all the fir.store operations, there's no read from the address.
// - Assume no address aliasing, no sharing
// - The only store operation is fir.store
//
// Why correct:
// - if no read between, write to the same address is idempotent.
struct ReduceRepeatedStore: public OpRewritePattern<fir::DoLoopOp> {
 using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(fir::DoLoopOp doLoopOp,
                                PatternRewriter &rewriter) const final {
    // key: address ssa, value: list of repeated storeOps
    // At last, except for the last storeOp (because we suppose it will be read after the loop!), all vectors can be erased.
    llvm::DenseMap<Value, llvm::SmallVector<fir::StoreOp>> repeatedStoreOps;
    doLoopOp.walk([&](Operation* op){
      llvm::TypeSwitch<Operation *>(op)
        .Case<fir::StoreOp>([&](fir::StoreOp storeOp){
          auto addr = storeOp.getMemref();
          auto opList = repeatedStoreOps.find(addr);
          if (opList != repeatedStoreOps.end()) {
            opList->getSecond().push_back(storeOp);  
          } else {
            llvm::SmallVector<fir::StoreOp> emptyList;
            emptyList.push_back(storeOp); 
            repeatedStoreOps[addr] = emptyList;
          }
          return;
        })
        .Case<fir::LoadOp, hlfir::DesignateOp, func::CallOp>([&](auto typedOp){
          auto popLast = [&](Value addr){
            auto it = repeatedStoreOps.find(addr);
            if (it != repeatedStoreOps.end() && !it->getSecond().empty()) {
              if (it->getSecond().size() == 1) {
                repeatedStoreOps.erase(addr);
              } else {
                it->getSecond().pop_back();
              }
            }
          };
          if (fir::LoadOp loadOp = llvm::dyn_cast<fir::LoadOp>(op)) {
            popLast(loadOp.getMemref());
          } else if (hlfir::DesignateOp desigOp = llvm::dyn_cast<hlfir::DesignateOp>(op)) {
            popLast(desigOp.getMemref());
          } else if (func::CallOp callOp = llvm::dyn_cast<func::CallOp>(op)) {
            llvm::for_each(callOp.getOperands(), [&](auto param){popLast(param);});
          } else {
            llvm::errs() << "Unexpected Op type!\n";
            std::exit(EXIT_FAILURE);
          }
          return;
        })
        .Default([](auto){return;});
      return;
    });
    llvm::DenseSet<fir::StoreOp> toDeleteOps;  
    for (auto& entry: repeatedStoreOps) {
      if (entry.getSecond().empty() || entry.getSecond().size() == 1) {
        continue;
      }
      entry.getSecond().pop_back();
      llvm::for_each(entry.getSecond(), [&](auto repeatedOp){toDeleteOps.insert(repeatedOp);});
    }
    auto deletedSize = toDeleteOps.size();
    for (auto toDeleteOp: toDeleteOps) {
      rewriter.eraseOp(toDeleteOp);
    }
    if (deletedSize > 0) {
      return success();
    }
    return failure();
  }
};

struct CleanFIRLoadPass : public mlir::PassWrapper<CleanFIRLoadPass, mlir::OperationPass<mlir::func::FuncOp>> {
   void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-clean-fir-load"; }

  void runOnOperation() override { 
    auto funcOp = getOperation();
    MLIRContext* ctx = getOperation()->getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ReduceLoadStoredAddress>(ctx);
    patterns.add<ReduceRepeatedStore>(ctx);
    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
  }
};
};

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createCleanFIRLoadPass() {
    return std::make_unique<CleanFIRLoadPass>();
  }   

  void registerCleanFIRLoadPass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createCleanFIRLoadPass();});
  }
}

