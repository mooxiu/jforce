#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
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
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include "../support/utilities.h"
#include "mlir/Transforms/Passes.h"

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
  llvm::SmallVector<fir::DoLoopOp, 4> dlOps;
  fop.walk([&](fir::DoLoopOp dlOp){
    if (dlOp.getNumRegionIterArgs() > 0) {
      assert(dlOp.getNumRegionIterArgs() == 1);
      dlOps.push_back(dlOp);
    }
  });

  for (auto dlOp: dlOps) {
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


// Jforce suppose the loop in included in
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


    // static DoLoopOp create(
    //  ::mlir::OpBuilder &builder, 
    //  ::mlir::Location location, 
    //  mlir::Value lowerBound, 
    //  mlir::Value upperBound, 
    //  mlir::Value step, 
    //  bool unordered = false, 
    //  bool finalCountValue = false, 
    //  mlir::ValueRange iterArgs = {}, 
    //  mlir::ValueRange reduceOperands = {}, 
    //  llvm::ArrayRef<mlir::Attribute> reduceAttrs = {}, 
    //  llvm::ArrayRef<mlir::NamedAttribute> attributes = {});

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
  StringRef getArgument() const override { return "jforce-clean-fir-loop"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());
    if (funcOp.walk([&](fir::DoLoopOp dlOp) { return WalkResult::interrupt(); })
            .wasInterrupted() == false) {
      // There is no do loop inside, just return
      DEBUG_PRINT("There's no do loop inside, skip CleanFIRLoopPass.");
      return;
    };
    // debugging(funcOp);
    cleanIterArgs(opBuilder, funcOp);
    cleanLoopResult(opBuilder, funcOp);
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
