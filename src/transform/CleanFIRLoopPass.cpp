#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <cstdlib>
#include "../support/utilities.h"

using namespace mlir;

namespace {

static void debugging(func::FuncOp fop) {
  fop.walk(
      [&](fir::DoLoopOp dlOp) { 
      // dlOp.getOperation()->print(llvm::dbgs()); 
      auto terminator = dlOp.getBody()->getTerminator();
      terminator->print(llvm::dbgs());
    });
};


static void cleanIterArgs(mlir::OpBuilder& opBuilder, func::FuncOp fop) {
  llvm::SmallVector<fir::DoLoopOp, 4> doLoopOps;
  fop.walk([&](fir::DoLoopOp dlOp){
    if (dlOp.getNumRegionIterArgs() > 0) {
      assert(dlOp.getNumRegionIterArgs() == 1);
      doLoopOps.push_back(dlOp);
    }
  });

  for (auto dlOp: doLoopOps) {
    assert(dlOp.getRegionIterArgs().size() == 1);
    // The real induction varaible IV
    auto loopIV = dlOp.getInductionVar();

    // Return %arg10 in `fir.do_loop ... iter_args(%arg10 = %9)`
    auto iterArg = dlOp.getRegionIterArgs()[0]; 
    Value storedIterArg;

    llvm::SmallVector<Operation*> toDelete; 
    dlOp.walk([&](Operation* op){
      llvm::TypeSwitch<Operation *>(op)
        .Case([&](fir::StoreOp storeOp){
          // storedIterArg is assigned with %0#0 in `fir.store %arg10 to %0#0`
          if (storeOp.getValue() == iterArg) {
            storedIterArg = storeOp.getMemref();
            toDelete.push_back(storeOp);
          }
        })
        .Case([&](fir::LoadOp loadOp){
          if (storedIterArg != nullptr && loadOp.getMemref() == storedIterArg){
            // %0#0 is where the iter_arg is stored to:
            // %12 = fir.load %0#0 : !fir.ref<i32>
            // We need to replace this with %12 = fir.convert IV: (index) -> get%12Type
            opBuilder.setInsertionPoint(loadOp); 
            auto convertOp = fir::ConvertOp::create(
              opBuilder, 
              loadOp.getLoc(), 
              loadOp.getResult().getType(),
              {loopIV}, 
              {}
            );
            loadOp.getResult().replaceAllUsesWith(convertOp.getResult());
            toDelete.push_back(loadOp);
          }
        });
    });
    for (auto* op: toDelete) {
      op->erase();
    }
  }

  return;
}


// example: 
//  %8 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
// Inducation Variable (IV): %arg9 which we should use.
// Iteration Arg: %arg10 which would be loaded and stored, we should avoid use it (them).
static void cleanIterArgsNew(mlir::OpBuilder& opBuilder, func::FuncOp fop) {
  llvm::SmallVector<fir::DoLoopOp> doLoops;
  fop.walk([&](fir::DoLoopOp doLoop){doLoops.push_back(doLoop);});    
  
  llvm::SmallVector<Operation*> toDeleteOps;
  for (auto doLoop: doLoops) {
    assert(doLoop.getRegionIterArgs().size() == 1);
    auto loopIV = doLoop.getInductionVar();
    auto iterArg = doLoop.getRegionIterArgs()[0]; 

    doLoop.walk([&](fir::StoreOp storeOp){
      if (storeOp.getValue() == iterArg){
        opBuilder.setInsertionPoint(storeOp);
        auto convertOp = fir::ConvertOp::create(opBuilder, storeOp.getLoc(), iterArg.getType(), loopIV, {});
        fir::StoreOp::create(opBuilder, storeOp.getLoc(), convertOp.getResult(), storeOp.getMemref());  
        toDeleteOps.push_back(storeOp);
      }
      return;
    });

    doLoop.walk([&](memref::StoreOp storeOp){
      if (storeOp.getValue() == iterArg){
        opBuilder.setInsertionPoint(storeOp);
        auto convertOp = fir::ConvertOp::create(opBuilder, storeOp.getLoc(), iterArg.getType(), loopIV, {});
        memref::StoreOp::create(opBuilder, storeOp.getLoc(), convertOp.getResult(), storeOp.getMemref(), storeOp.getIndices());
        toDeleteOps.push_back(storeOp);
      }
      return;
    });
  }

  for (auto* toDeleteOp: toDeleteOps){
    toDeleteOp->erase();
  }
}


// Jforce suppose the loop in included in.
// Should consider whether this is a first private.
static void cleanLoopResult(OpBuilder& opBuilder, func::FuncOp fop) {
  llvm::SmallVector<fir::DoLoopOp, 4> oldDoLoopOps;
  fop.walk([&](fir::DoLoopOp dlOp){
    // Delete the result and reference to the do loop result
    if (dlOp.getNumResults() == 0) {
      return;
    }
    oldDoLoopOps.push_back(dlOp);
  });

  for (auto dlOp: oldDoLoopOps) {
    Value res = dlOp.getResult(0);


    llvm::SmallVector<Operation*> toDelete;
    auto users = res.getUsers();
    for (auto user: users) {
      auto storeOp = llvm::dyn_cast<fir::StoreOp>(user);
      assert(storeOp);
      toDelete.push_back(storeOp);
    }
    for (auto op: toDelete) {
      op->erase();
    }

    // Delete the return valus of do loop
    opBuilder.setInsertionPoint(dlOp);
    mlir::NamedAttrList filteredAttrs(dlOp->getAttrs());
    filteredAttrs.erase("unordered");
    filteredAttrs.erase("finalCountValue");
    filteredAttrs.erase(dlOp.getOperandSegmentSizeAttr());
    auto newDoLoopOp = fir::DoLoopOp::create(
      opBuilder, 
      dlOp.getLoc(), 
      dlOp.getLowerBound(), 
      dlOp.getUpperBound(),
      dlOp.getStep(),
      dlOp.getUnordered().has_value()? dlOp.getUnordered().value(): false,
      /*finalCountValue*/false,
      /*ArgIters*/ {},
      {},
      {},
      filteredAttrs
    );


    dlOp.getInductionVar().replaceAllUsesWith(newDoLoopOp.getInductionVar());


    auto newBlock = newDoLoopOp.getBody(); 
    auto existingTerminator = newBlock->getTerminator();
    newBlock->getOperations().splice(
      Block::iterator(existingTerminator),
      dlOp.getBody()->getOperations(),
      dlOp.getBody()->getOperations().begin(),
      Block::iterator(dlOp.getBody()->getTerminator())
    );

    assert(dlOp.use_empty());
    dlOp.erase();
  }

  return;
};

struct CleanFIRLoopPass
    : public PassWrapper<CleanFIRLoopPass, OperationPass<func::FuncOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-clean-fir-loop"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());
    cleanIterArgsNew(opBuilder, funcOp);
    // cleanLoopResult(opBuilder, funcOp);
  };
};
} // namespace

namespace xla_jit {
std::unique_ptr<mlir::Pass> createCleanFIRLoopPass() {
  return std::make_unique<CleanFIRLoopPass>();
}

void registerCleanFIRLoopPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createCleanFIRLoopPass(); });
};
} // namespace xla_jit
