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
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <cstdlib>
#include <optional>

using namespace mlir;

namespace {
enum StoreOpType {
  FIR_STORE,
  MEMREF_STORE
};

struct IterArgsBuffer {
  std::optional<StoreOpType> StoreInst;
  Value Mem;
};

IterArgsBuffer buffer;

// example: 
//  %8 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
// Inducation Variable (IV): %arg9 which we should use.
// Iteration Arg: %arg10 which would be loaded and stored, we should avoid use it (them).
static void replaceIterArgs(mlir::OpBuilder& opBuilder, fir::DoLoopOp doLoop) {
  llvm::SmallVector<Operation*> toDeleteOps;
  auto loopIV = doLoop.getInductionVar();
  assert(doLoop.getRegionIterArgs().size() == 1);
  auto iterArg = doLoop.getRegionIterArgs()[0];

  doLoop.walk([&](fir::StoreOp storeOp){
    if (storeOp.getValue() == iterArg){
      buffer.StoreInst = StoreOpType::FIR_STORE;
      buffer.Mem = storeOp.getMemref();

      opBuilder.setInsertionPoint(storeOp);
      auto convertOp = fir::ConvertOp::create(opBuilder, storeOp.getLoc(), iterArg.getType(), loopIV, {});
      fir::StoreOp::create(opBuilder, storeOp.getLoc(), convertOp.getResult(), storeOp.getMemref());  
      toDeleteOps.push_back(storeOp);
    }
    return;
  });

  doLoop.walk([&](memref::StoreOp storeOp){
    if (storeOp.getValue() == iterArg){
      buffer.StoreInst = StoreOpType::MEMREF_STORE;
      buffer.Mem = storeOp.getMemref();

      opBuilder.setInsertionPoint(storeOp);
      auto convertOp = fir::ConvertOp::create(opBuilder, storeOp.getLoc(), iterArg.getType(), loopIV, {});
      memref::StoreOp::create(opBuilder, storeOp.getLoc(), convertOp.getResult(), storeOp.getMemref(), storeOp.getIndices());
      toDeleteOps.push_back(storeOp);
    }
    return;
  });

  for (auto* toDeleteOp: toDeleteOps){
    toDeleteOp->erase();
  }

  if (!iterArg.use_empty()) {
    opBuilder.setInsertionPointToStart(doLoop.getBody());
    auto convertOp = fir::ConvertOp::create(
        opBuilder, doLoop.getLoc(), iterArg.getType(), loopIV, {});
    iterArg.replaceAllUsesWith(convertOp.getResult());
  }
}

static void replaceLoopSignature(OpBuilder& opBuilder, fir::DoLoopOp doLoop){
  opBuilder.setInsertionPoint(doLoop);
  mlir::NamedAttrList filteredAttrs(doLoop->getAttrs());
  filteredAttrs.erase("unordered");
  filteredAttrs.erase("finalCountValue");
  filteredAttrs.erase(doLoop.getOperandSegmentSizeAttr());
  auto newDoLoop = fir::DoLoopOp::create(
    opBuilder, 
    doLoop.getLoc(), 
    doLoop.getLowerBound(), 
    doLoop.getUpperBound(),
    doLoop.getStep(),
    doLoop.getUnordered().has_value()? doLoop.getUnordered().value(): false,
    /*finalCountValue*/ doLoop.getFinalValue().has_value(),
    /*ArgIters*/ {},
    /*reduceOperands*/ {},
    /*reduceAttrs=*/ {},
    filteredAttrs
  );
  doLoop.getInductionVar().replaceAllUsesWith(newDoLoop.getInductionVar());
  auto newBlock = newDoLoop.getBody(); 
  auto existingTerminator = newBlock->getTerminator();
  newBlock->getOperations().splice(
    Block::iterator(existingTerminator),
    doLoop.getBody()->getOperations(),
    doLoop.getBody()->getOperations().begin(),
    Block::iterator(doLoop.getBody()->getTerminator())
  );

  auto oldTerminator = cast<fir::ResultOp>(doLoop.getBody()->getTerminator());
  assert(oldTerminator.getNumOperands() == 1);
  assert(doLoop.getNumResults() == 1);
  if (!doLoop.getResult(0).use_empty()) {
    assert(buffer.StoreInst.has_value() &&
         "Loop result is used, but no backing store was found");
    assert(buffer.Mem &&
         "Loop result is used, but no backing memory was found"); 
    Value newResult;
    switch (*buffer.StoreInst) {
      case StoreOpType::FIR_STORE:
        opBuilder.setInsertionPoint(existingTerminator);
        fir::StoreOp::create(opBuilder, oldTerminator.getLoc(), oldTerminator.getOperand(0), buffer.Mem);
        opBuilder.setInsertionPointAfter(newDoLoop);
        newResult = fir::LoadOp::create(opBuilder, newDoLoop.getLoc(), buffer.Mem).getResult();
        break;
      case StoreOpType::MEMREF_STORE:
        opBuilder.setInsertionPoint(existingTerminator);
        memref::StoreOp::create(opBuilder, oldTerminator.getLoc(), oldTerminator.getOperand(0), buffer.Mem, {});
        opBuilder.setInsertionPointAfter(newDoLoop);
        newResult = memref::LoadOp::create(opBuilder, newDoLoop.getLoc(), buffer.Mem, {}).getResult();
        break;
    }
    doLoop.getResult(0).replaceAllUsesWith(newResult);
  }
  oldTerminator.erase();
  assert(doLoop.use_empty());
  doLoop.erase();
}

struct CleanFIRLoopPass
    : public PassWrapper<CleanFIRLoopPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CleanFIRLoopPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-clean-fir-loop"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp.getContext());
    
    llvm::SmallVector<fir::DoLoopOp> doLoops;
    funcOp.walk([&](fir::DoLoopOp doLoop){doLoops.push_back(doLoop);});    
    for (auto doLoop: doLoops) {
      if (doLoop.getNumRegionIterArgs() > 0) {
        assert(doLoop.getNumRegionIterArgs() == 1 && "Currently only deal with one iterarg");
        buffer = {};
        replaceIterArgs(opBuilder, doLoop);
        replaceLoopSignature(opBuilder, doLoop);
      }
    } 
    funcOp.dump();
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
