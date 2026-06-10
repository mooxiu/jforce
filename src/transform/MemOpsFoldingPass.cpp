/// INFO: the objective of this is to conclude the common part of `OptimizingMemOps` and `IfConversion`. 


#include "Utils.h"
#include "MemUtils.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include <optional>
#include <type_traits>

using namespace mlir;

namespace {

// TODO: this is not used for now, but for more aggressive optimization, 
// we should use this instead of the memref itself.
struct MemoryLocation {
  Value base;
  SmallVector<Value> indices;

  bool isExactly(const MemoryLocation& other) const {
    if (base != other.base) return false;
    if (indices.size() != other.indices.size()) return false;
    for (auto [idx1, idx2] : llvm::zip(indices, other.indices)) {
      if (idx1 != idx2) return false;
    }
    return true;
  }
};

static std::optional<Value> __getAncient(Value val) {
  if (!llvm::isa<fir::ReferenceType, MemRefType, fir::BoxType>(val.getType())) {
    llvm::errs() << "This is not a address\n";
    return std::nullopt;
  }

  auto defOp = val.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  if (llvm::isa<fir::DeclareOp, hlfir::DeclareOp, fir::AllocaOp, memref::AllocaOp>(defOp)) {
    return val;
  } else if (auto designateOp = llvm::dyn_cast<hlfir::DesignateOp>(defOp)) {
    return __getAncient(designateOp.getMemref());
  } else if (auto convertOp = llvm::dyn_cast<fir::ConvertOp>(defOp)) {
    return __getAncient(convertOp.getOperand());
  }
  llvm::errs() << "\n defOp: ";
  defOp->print(llvm::errs());
  llvm::errs() << "\n";
  llvm_unreachable("Should have returned before.\n");
  return std::nullopt;
}

static std::optional<bool> isTargetOrPointer(Value val) {
  std::optional<Value> ancientVal = __getAncient(val);
  if (!ancientVal.has_value()) {
    return std::nullopt;
  }
  auto defOp = ancientVal->getDefiningOp();
  assert(
    llvm::isa<fir::DeclareOp>(defOp) ||
    llvm::isa<hlfir::DeclareOp>(defOp) ||
    llvm::isa<fir::AllocaOp>(defOp) ||
    llvm::isa<memref::AllocaOp>(defOp));

  ::std::optional<::fir::FortranVariableFlagsEnum> fortranAttrs;
  if (auto firDeclareOp = llvm::dyn_cast<fir::DeclareOp>(defOp)) {
    fortranAttrs = firDeclareOp.getFortranAttrs();
  } else if (auto hlfirDeclareOp = llvm::dyn_cast<hlfir::DeclareOp>(defOp)) {
    fortranAttrs = hlfirDeclareOp.getFortranAttrs();
  } else if (llvm::isa<fir::AllocaOp, memref::AllocaOp>(defOp)) {
    return false;
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
//
// static llvm::SmallVector<Value> getReadFromAddr(Operation *op) {
//   llvm::SmallVector<Value> addrs;
//   if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(op)) {
//     addrs.push_back(loadOp.getMemref());
//   } else if (auto desigOp = llvm::dyn_cast<hlfir::DesignateOp>(op)) {
//     addrs.push_back(desigOp.getMemref());
//   } else if (auto callOp = llvm::dyn_cast<func::CallOp>(op)) {
//     for (auto param : callOp.getOperands()) {
//       addrs.push_back(param);
//     }
//   }
//   return addrs;
// }

static bool switchableVals(Value v1, Value v2) {
  auto v1TyInfo = inspectTypeInfo(v1.getType());
  auto v2TyInfo = inspectTypeInfo(v2.getType());
  return v1TyInfo.shape.equals(v2TyInfo.shape) 
    && v1TyInfo.elementTy==v2TyInfo.elementTy;
}

// The mem we're going to assign to
static mlir::Value getLHS(mlir::Operation *op) {
  if (auto store = llvm::dyn_cast<fir::StoreOp>(op))
    return store.getMemref();
  if (auto assign = llvm::dyn_cast<hlfir::AssignOp>(op))
    return assign.getLhs();
  return nullptr;
}

// The value we're going to assign
static mlir::Value getRHS(mlir::Operation *op) {
  if (auto store = llvm::dyn_cast<fir::StoreOp>(op))
    return store.getValue();
  if (auto assign = llvm::dyn_cast<hlfir::AssignOp>(op))
    return assign.getRhs();
  return nullptr;
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
// TODO: rename to `FoldReadFromWriteMem`
template <typename LoadTy, typename StoreTy>
struct ReduceWriteAndReadSameAddr : public OpRewritePattern<LoadTy> {
  using OpRewritePattern<LoadTy>::OpRewritePattern;

  // FIXME: do alias analysis
  static bool checkFoldingSafety(Value addr) {
    // Target or pointer can be aliased, here we conservatively stop doing this optimization.
    if (isTargetOrPointer(addr).has_value() && isTargetOrPointer(addr).value()) {
      return false;
    }
    return true;
  }


  LogicalResult matchAndRewrite(LoadTy loadOp,
                                PatternRewriter &rewriter) const final {
    auto addr = loadOp.getMemref();
    if (!checkFoldingSafety(addr)) {
      return failure();
    }
 
    Operation *currOp = loadOp;
    while ((currOp = currOp->getPrevNode())) {
      auto isReplaced = false;
      auto shouldBreak = false;
      llvm::TypeSwitch<Operation *>(currOp)
        .Case<StoreTy>([&](auto storeOp) {
          if (getLHS(storeOp) == addr) {
            if (switchableVals(getRHS(storeOp), loadOp.getResult())) {
              rewriter.replaceOp(loadOp, getRHS(storeOp));
              isReplaced = true;
              return;
            } else {
              // Something has been written to the same mem, but we cannot replace.
              shouldBreak = true;
              return;
            }
          }
        })
        .template Case<func::ReturnOp>([&](auto) { 
          shouldBreak = true;
          return;
        })
        .Default([&](Operation* op) {
          if (isOperationPossiblelyWriteToAddr(op, addr)) {
            shouldBreak = true;
          }
          // INFO: cover cases like loop, if ... but over conservative which is fine.
          if (op->getNumRegions() > 0) {
            shouldBreak = true;
          }
          return;
        });
      if (isReplaced) return success();
      if (shouldBreak) return failure();
    }
    return failure();
  };
};



template<typename T, typename = void>
struct has_get_indices: std::false_type {};

template<typename T>
struct has_get_indices<T, std::void_t<decltype(std::declval<T>().getIndices())>>: std::true_type {};

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
// TODO: rename to `FoldStoreToReadMem`
template<typename StoreTy, typename LoadTy>
struct ReduceReadAndWriteSameAddr : public OpRewritePattern<StoreTy> {
  using OpRewritePattern<StoreTy>::OpRewritePattern;

  // FIXME: do alias analysis
  static bool checkFoldingSafety(Value addr) {
    // Target or pointer can be aliased, here we conservatively stop doing this optimization.
    if (isTargetOrPointer(addr).has_value() && isTargetOrPointer(addr).value()) {
      return false;
    }
    return true;
  }

  
  LogicalResult matchAndRewrite(StoreTy storeOp,
                                PatternRewriter &rewriter) const final {
    Value addr = storeOp.getMemref();
    if (!checkFoldingSafety(addr)) return failure();

    Value val = storeOp.getValue();
    Operation* valDefineOp = val.getDefiningOp();
    if (!valDefineOp || valDefineOp->getBlock() != storeOp->getBlock()) {
      // We do not consider cross block situation, it will makes the analysis much more difficult.
      return failure();
    }

    // Require the load address and the store address, load value and read value be the same.
    bool isReadFromSameMem = false;
    llvm::TypeSwitch<Operation*>(valDefineOp)
      .Case<LoadTy>([&](LoadTy loadOp){
        if (loadOp.getMemref() != addr) return;
        if constexpr (has_get_indices<LoadTy>::value) {
          if constexpr (has_get_indices<StoreTy>::value) {
            if (loadOp.getIndices() != storeOp.getIndices()) return;
          } else {
            if (!loadOp.getIndices().empty()) return;
          }
        } else {
          if constexpr (has_get_indices<StoreTy>::value) {
            if (!storeOp.getIndices().empty()) return;
          }
        }
        if (loadOp.getResult() != val) return;
        isReadFromSameMem = true;
        return;
      })
      .Default([](auto){return;});
    if (!isReadFromSameMem) {
      return failure();
    }

    for (Operation* op = valDefineOp->getNextNode(); op != storeOp; op = op->getNextNode()) {
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

    for (auto &op: doLoopOp.getRegion().front().getOperations()) {
      llvm::TypeSwitch<Operation&>(op)
        .template Case<fir::StoreOp, memref::StoreOp>(
          [&](auto storeOp) {
            auto addr = storeOp.getMemref();
            auto opList = repeatedStoreOps.find(addr);
            if (opList != repeatedStoreOps.end()) {
              opList->getSecond().push_back(storeOp);
            } else {
              llvm::SmallVector<Operation *> emptyList{storeOp};
              repeatedStoreOps[addr] = emptyList;
            }
            return;
          }
        )
        .template Case<fir::LoadOp, hlfir::DesignateOp, memref::LoadOp>(
          [&](auto loadOp) {
            auto addr = loadOp.getMemref();
            popLast(addr);
            return;
          }
        )
        .template Case<func::CallOp>(
          [&](func::CallOp callOp) {
            llvm::for_each(callOp.getOperands(), [&](auto param) { popLast(param); });
            return;
          }
        )
        .Default(
          [&](auto& op) { 
            if (!op.getRegions().empty()) {
              // conservative move: suppose there is read of all operations here.
              for (auto& entry: repeatedStoreOps) {
                popLast(entry.getFirst());
              }
            }
            return; 
          }
        );
    }

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

struct MemOpsFoldingPass
    : public mlir::PassWrapper<MemOpsFoldingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-mem-ops-folding"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = getOperation()->getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ReduceWriteAndReadSameAddr<fir::LoadOp, hlfir::AssignOp>>(ctx);
    patterns.add<ReduceWriteAndReadSameAddr<fir::LoadOp, fir::StoreOp>>(ctx);
    patterns.add<ReduceWriteAndReadSameAddr<memref::LoadOp, memref::StoreOp>>(ctx);
    patterns.add<ReduceReadAndWriteSameAddr<fir::StoreOp, fir::LoadOp>>(ctx);
    patterns.add<ReduceReadAndWriteSameAddr<memref::StoreOp, memref::LoadOp>>(ctx);
    patterns.add<ReduceRepeatWriteAddr<fir::DoLoopOp>>(ctx);
    patterns.add<ReduceRepeatWriteAddr<scf::ForOp>>(ctx);
    patterns.add<ReduceRepeatWriteAddr<affine::AffineForOp>>(ctx);
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
std::unique_ptr<mlir::Pass> createMemOpsFoldingPass() {
  return std::make_unique<MemOpsFoldingPass>();
}

void registerMemOpsFoldingPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createMemOpsFoldingPass(); });
}
} // namespace xla_jit
