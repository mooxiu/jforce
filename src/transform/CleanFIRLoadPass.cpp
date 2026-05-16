#include "../support/utilities.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include <optional>

using namespace mlir;

namespace {

static std::optional<Value> __getAncient(Value val) {
  DEBUG_PRINT_VAL(val);

  if (!llvm::isa<fir::ReferenceType, MemRefType>(val.getType())) {
    llvm::errs() << "This is not a address\n";
    return std::nullopt;
  }

  auto defOp = val.getDefiningOp();
  if (llvm::isa<fir::DeclareOp, hlfir::DeclareOp>(defOp)) {
    return val;
  } else if (auto convertOp = llvm::dyn_cast<fir::ConvertOp>(defOp)) {
    return __getAncient(convertOp.getOperand());
  }
  llvm_unreachable("Should have returned before.\n");
  return std::nullopt;
}

static std::optional<bool> isTargetOrPointer(Value val) {
  std::optional<Value> ancientVal = __getAncient(val);
  if (!ancientVal.has_value()) {
    return std::nullopt;
  }
  auto defOp = ancientVal->getDefiningOp();
  assert(llvm::isa<fir::DeclareOp>(defOp) ||
         llvm::isa<hlfir::DeclareOp>(defOp));

  ::std::optional<::fir::FortranVariableFlagsEnum> fortranAttrs;
  if (auto firDeclareOp = llvm::dyn_cast<fir::DeclareOp>(defOp)) {
    fortranAttrs = firDeclareOp.getFortranAttrs();
  } else if (auto hlfirDeclareOp = llvm::dyn_cast<hlfir::DeclareOp>(defOp)) {
    fortranAttrs = hlfirDeclareOp.getFortranAttrs();
  } else {
    llvm_unreachable("Should be one of the above declareOps!\n");
  }
  if (fortranAttrs.has_value()) {
    auto attr = fortranAttrs.value();
    return fir::bitEnumContainsAny(attr,
                                   fir::FortranVariableFlagsEnum::target) ||
           fir::bitEnumContainsAny(attr,
                                   fir::FortranVariableFlagsEnum::pointer);
  } else {
    return false;
  }
};

static llvm::SmallVector<Value> getReadFromAddr(Operation *op) {
  llvm::SmallVector<Value> addrs;
  if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(op)) {
    addrs.push_back(loadOp.getMemref());
  } else if (auto desigOp = llvm::dyn_cast<hlfir::DesignateOp>(op)) {
    addrs.push_back(desigOp.getMemref());
  } else if (auto callOp = llvm::dyn_cast<func::CallOp>(op)) {
    for (auto param : callOp.getOperands()) {
      addrs.push_back(param);
    }
  }
  return addrs;
}

static bool isOperationPossiblelyWriteToAddr(Operation* op, Value addr) {
  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memInterface) {
    return llvm::is_contained(op->getOperands(), addr);
  }
  
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memInterface.getEffects(effects);
  for (const auto &effect : effects) {
    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      Value effectValue = effect.getValue();
      if (effectValue == addr || effectValue == nullptr) {
        return true;
      }
    }
  }
  return false;
}

static bool isOperationPossiblelyReadFromAddr(Operation* op, Value addr) {
  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memInterface) {
    return llvm::is_contained(op->getOperands(), addr);
  }
  
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memInterface.getEffects(effects);
  for (const auto &effect : effects) {
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      Value effectValue = effect.getValue();
      if (effectValue == addr || effectValue == nullptr) {
        return true;
      }
    }
  }
  return false;
}

// Mem2reg is supposed to cover alloca, but we need aliasing analysis here.
//
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
// Notice that we do not erase the store operation as it can be DCE-ed if it is
// not been used.
template <typename LoadTy>
struct ReduceWriteAndReadSameAddr : public OpRewritePattern<LoadTy> {
  using OpRewritePattern<LoadTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadTy loadOp,
                                PatternRewriter &rewriter) const final {
    auto addr = loadOp.getMemref();
    // Target or pointer can be aliased, here we conservatively stop doing this
    // optimization.
    if (isTargetOrPointer(addr).has_value() && isTargetOrPointer(addr).value()) {
      return failure();
    }
    Operation *currOp = loadOp;
    while ((currOp = currOp->getPrevNode())) {
      auto isReplaced = false;
      auto isBreaked = false;
      llvm::TypeSwitch<Operation *>(currOp)
          .template Case<memref::StoreOp, fir::StoreOp>([&](auto storeOp) {
            if (storeOp.getMemref() == addr) {
              rewriter.replaceOp(loadOp, storeOp.getValue());
              isReplaced = true;
            }
          })
          .template Case<func::ReturnOp>(
              [&](func::ReturnOp retOp) { isBreaked = true; })
          .Default([](auto) {});
      if (isReplaced) {
        return success();
      }
      if (isBreaked) {
        return failure();
      }
    }
    return failure();
  };
};

// Before:
//   %13 = memref.load %8[] : memref<i32>
//   memref.store %13, %8[] : memref<i32>
// After:
//    (only delete the store, and hope the load will be erased in other pattern matchings)
//   %13 = memref.load %8[] : memref<i32>
//
// Precondition:
//   - address should be non aliasing
//   - between the load and store, there should not be any other store to the addr!
template<typename StoreTy>
struct ReduceReadAndWriteSameAddr : public OpRewritePattern<StoreTy> {
  using OpRewritePattern<StoreTy>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(StoreTy storeOp,
                                PatternRewriter &rewriter) const final {
    Value addr = storeOp.getMemref();
    if (isTargetOrPointer(addr).has_value() && isTargetOrPointer(addr).value()) {
      return failure();
    }

    Value val = storeOp.getValue();
    Operation* readOp = val.getDefiningOp();
    if (readOp->getBlock() != storeOp->getBlock()) {
      // We do not consider cross block situation, it will makes the analysis much more difficult.
      return failure();
    }

    // Require the load address and the store address, load value and read value be the same.
    if (auto memLoad = llvm::dyn_cast<memref::LoadOp>(readOp)) {
      if (memLoad.getMemref() != addr || memLoad.getResult() != val) {
        return failure();
      }
    } else if (auto firLoad = llvm::dyn_cast<fir::LoadOp>(readOp)) {
      if (firLoad.getMemref() != addr || firLoad.getResult() != val) {
        return failure();
      }
    } else {
      DEBUG_PRINT("Unexpected readOp: ");
      DEBUG_PRINT_OP(readOp);
      return failure();
    }

    for (Operation* op = readOp->getNextNode(); op != storeOp; op = op->getNextNode()) {
      if (isOperationPossiblelyWriteToAddr(op, addr)) {
        return failure();
      }
    }

    rewriter.eraseOp(storeOp);
    return success();
  }
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
template <typename LoopTy>
struct ReduceRepeatWriteAddr : public OpRewritePattern<LoopTy> {
  using OpRewritePattern<LoopTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(LoopTy doLoopOp,
                                PatternRewriter &rewriter) const final {
    // key: address ssa, value: list of repeated storeOps
    // At last, except for the last storeOp (because we suppose it will be read
    // after the loop!), all vectors can be erased.
    llvm::DenseMap<Value, llvm::SmallVector<Operation *>> repeatedStoreOps;

    auto popLast = [&](Value addr) {
      if (isTargetOrPointer(addr).has_value() && isTargetOrPointer(addr).value()) {
        // DO NOTHING
        return;
      }
      auto it = repeatedStoreOps.find(addr);
      if (it != repeatedStoreOps.end() && !it->getSecond().empty()) {
        if (it->getSecond().size() == 1) {
          repeatedStoreOps.erase(addr);
        } else {
          it->getSecond().pop_back();
        }
      }
    };

    doLoopOp.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .template Case<fir::StoreOp, memref::StoreOp>([&](auto storeOp) {
            auto addr = storeOp.getMemref();
            auto opList = repeatedStoreOps.find(addr);
            if (opList != repeatedStoreOps.end()) {
              opList->getSecond().push_back(storeOp);
            } else {
              llvm::SmallVector<Operation *> emptyList{storeOp};
              repeatedStoreOps[addr] = emptyList;
            }
            return;
          })
          .template Case<fir::LoadOp, hlfir::DesignateOp, memref::LoadOp>(
              [&](auto loadOp) {
                auto addr = loadOp.getMemref();
                popLast(addr);
                return;
              })
          .template Case<func::CallOp>([&](func::CallOp callOp) {
            llvm::for_each(callOp.getOperands(),
                           [&](auto param) { popLast(param); });
            return;
          })
          .Default([](auto) { return; });
      return;
    });

    llvm::DenseSet<Operation *> toDeleteOps;
    for (auto &entry : repeatedStoreOps) {
      if (entry.getSecond().empty() || entry.getSecond().size() == 1) {
        continue;
      }
      entry.getSecond().pop_back();
      llvm::for_each(entry.getSecond(),
                     [&](auto repeatedOp) { toDeleteOps.insert(repeatedOp); });
    }
    auto deletedSize = toDeleteOps.size();
    for (auto toDeleteOp : toDeleteOps) {
      rewriter.eraseOp(toDeleteOp);
    }
    if (deletedSize > 0) {
      return success();
    }
    return failure();
  }
};

// After the above patterns, now inside of doloop, it should be:
// "ReadOp* (StoreOp ReadOp)* StoreOp?"
// If there's only a `StoreOp` for an address in this DoLoop, we have the chance
// to hoist it out of the doLoop.
//
// Before:
//  DOLOOP{
//    ...
//    fir.store %18 to %0#0
//    ...
//  }
//
// After:
//  DOLOOP{
//    ...
//  }
//  fir.store ??? to %0#0
//
// Precondition:
// - no aliasing and sharing of %0#0
// - an address, there should only be one storeOp for that
// - no goto or return or other control operation in the loop
//
struct HoistDoLoopStoreOp : public OpRewritePattern<fir::DoLoopOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(fir::DoLoopOp doLoopOp,
                                PatternRewriter &rewriter) const final {
    llvm::DenseMap<Value, fir::StoreOp> storeOps;
    llvm::DenseMap<Value, llvm::SmallVector<bool>>
        addrOpStateMachine; // storeOp -> true, readOp -> flase
    doLoopOp.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<fir::LoadOp, hlfir::DesignateOp, func::CallOp>(
              [&](Operation *typedOp) {
                auto readAddrs = getReadFromAddr(typedOp);
                for (auto addr : readAddrs) {
                  auto it = addrOpStateMachine.find(addr);
                  if (it != addrOpStateMachine.end()) {
                    it->getSecond().push_back(false);
                  } else {
                    addrOpStateMachine[addr] = {false};
                  }
                }
              })
          .Case<fir::StoreOp>([&](fir::StoreOp storeOp) {
            auto addr = storeOp.getMemref();
            storeOps[addr] = storeOp;
            auto it = addrOpStateMachine.find(addr);
            if (it != addrOpStateMachine.end()) {
              it->getSecond().push_back(true);
            } else {
              addrOpStateMachine[addr] = {true};
            }
          })
          .Default([](auto) {});
    });

    llvm::DenseMap<Value, fir::StoreOp> toHoistStoreOps;
    llvm::DenseMap<Value, Value> toHoistStoredValues;
    for (const auto &entry : addrOpStateMachine) {
      // Have only one addr operation and it is a store
      if (entry.getSecond().size() == 1 && entry.getSecond()[0]) {
        auto addr = entry.getFirst();
        assert(storeOps.find(addr) != storeOps.end());
        auto internalStoreOp = storeOps[addr];
        toHoistStoreOps[addr] = internalStoreOp;
        // Do back track about the value, should store in where?
      } else {
        continue;
      }
    }

    rewriter.setInsertionPointAfter(doLoopOp);
    return failure();
  }
};


// Before:
//    // no write to %addr
//    loop {
//      ... no write to %addr
//      val = read %addr
//      ... no write to %addr
//    }
//
// After:
//    val = read %addr
//    loop {
//      ... no write to %addr
//    }
// 
// Precondition:
// - addr is not target or pointer, so no aliasing problem
// - in the loop, there's no possible write to %addr
// - addr and indices are defined outside of the loop
template<typename ReadOpTy>
struct HoistReadOpFromLoop : public OpRewritePattern<ReadOpTy> {
  using OpRewritePattern<ReadOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReadOpTy readOp,
                                PatternRewriter &rewriter) const final {
    auto loopOp = readOp->template getParentOfType<LoopLikeOpInterface>();
    if (!loopOp) {
      return failure();
    }

    Value addr = readOp.getMemref();
    if (!loopOp.isDefinedOutsideOfLoop(addr)) {
      return failure();
    }

    for (Value operand : readOp->getOperands()) {
      if (!loopOp.isDefinedOutsideOfLoop(operand)) {
        return failure();       
      }
    }

    bool hasWrite = false;
    loopOp->walk([&](Operation* op){
      if (isOperationPossiblelyWriteToAddr(op, addr)) {
        hasWrite = true;
        return WalkResult::interrupt();
      };
      return WalkResult::advance();
    });
    if (hasWrite) {
      return failure();
    }

    rewriter.moveOpBefore(readOp, loopOp);
    return success();
  }
};

struct CleanFIRLoadPass
    : public mlir::PassWrapper<CleanFIRLoadPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-clean-fir-load"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = getOperation()->getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ReduceWriteAndReadSameAddr<fir::LoadOp>>(ctx);
    patterns.add<ReduceWriteAndReadSameAddr<memref::LoadOp>>(ctx);
    patterns.add<ReduceReadAndWriteSameAddr<fir::StoreOp>>(ctx);
    patterns.add<ReduceReadAndWriteSameAddr<memref::StoreOp>>(ctx);
    patterns.add<ReduceRepeatWriteAddr<fir::DoLoopOp>>(ctx);
    patterns.add<ReduceRepeatWriteAddr<scf::ForOp>>(ctx);
    patterns.add<ReduceRepeatWriteAddr<affine::AffineForOp>>(ctx);
    patterns.add<HoistReadOpFromLoop<fir::LoadOp>>(ctx);
    patterns.add<HoistReadOpFromLoop<memref::LoadOp>>(ctx);
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
std::unique_ptr<mlir::Pass> createCleanFIRLoadPass() {
  return std::make_unique<CleanFIRLoadPass>();
}

void registerCleanFIRLoadPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createCleanFIRLoadPass(); });
}
} // namespace xla_jit
