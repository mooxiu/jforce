#include "../support/utilities.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <optional>

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
    DEBUG_PRINT("compare reading and storeOp");
    DEBUG_PRINT_OP(storeOp);
    DEBUG_PRINT_OP(readOp);
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
// If there's only a `StoreOp` for an addressxindices in this DoLoop, we have the chance
// to hoist it out of the doLoop.
//
// Before:
//  DOLOOP{
//    ...
//    write %val to addr[indices]
//    ...
//  }
//
// After:
//  DOLOOP{
//    ...
//  }
//  write %val to addr[indices]
//
// Precondition:
// - no aliasing and sharing of memref
// - there should be no other access to the same address (memref x indices)
// - the indice should not rely on IV.
//  - thinking about this: 
//      store %val1 to addr1[1]
//      store %val2 to addr1[2]
//    we can actually hoist both of them out of the loop!
// - no goto or return or other control operation in the loop: no break!
//
template<typename WriteOpTy>
struct HoistWriteOpFromLoop : public OpRewritePattern<WriteOpTy> {
  using OpRewritePattern<WriteOpTy>::OpRewritePattern;

  static Value getIV(Operation* loopOp) {
    if (auto firLoopOp = llvm::dyn_cast<fir::DoLoopOp>(loopOp)) {
      return firLoopOp.getInductionVar();
    } else if (auto scfForOp = llvm::dyn_cast<scf::ForOp>(loopOp)) {
      return scfForOp.getInductionVar();
    } else if (auto affineForOp = llvm::dyn_cast<affine::AffineForOp>(loopOp)) {
      return affineForOp.getInductionVar();
    } else {
      llvm::errs() << "Unexpected loopOp type, cannot get IV!\n";
      DEBUG_PRINT_OP(loopOp);
      return nullptr;
    }
  }

  // Record all the operations in the loop required to construct for the value
  static std::optional<llvm::SmallVector<Operation*>> trackValueInLoop(mlir::LoopLikeOpInterface loopOp, Value IV, Value val) {
    llvm::SmallSetVector<Value, 0> operandsToVisit;
    llvm::DenseSet<Value> visitedOperands{IV};
    llvm::SmallVector<Operation *> valTrack;    
    operandsToVisit.insert(val);
    while (operandsToVisit.size() > 0) {
      Value currOp = operandsToVisit.pop_back_val();
      visitedOperands.insert(currOp);
      Operation* defOp = currOp.getDefiningOp();
      // defOp is defined inside of loopOp
      auto operands = defOp -> getOperands(); 
      if (llvm::isa<memref::LoadOp, fir::LoadOp>(defOp)) {
        // Value of definition operation should not be a loadOp!
        return std::nullopt;
      }
      auto shouldTrackDefOp = false;
      for (auto operand : operands) {
        if (loopOp.isDefinedOutsideOfLoop(operand)) {
          continue;
        } 
        if (!visitedOperands.contains(operand)) {
          operandsToVisit.insert(operand);
          shouldTrackDefOp = true;
        }
      }
      if (shouldTrackDefOp) {
        valTrack.push_back(defOp);
      }
    }
    return valTrack;
  }

  static std::optional<llvm::SmallVector<Operation*>> trackIdxInLoop(mlir::LoopLikeOpInterface loopOp, Value IV, Value val) {
    if (val == IV) {
      return std::nullopt;
    }
    llvm::SmallSetVector<Value, 0> operandsToVisit;
    llvm::DenseSet<Value> visitedOperands{};
    llvm::SmallVector<Operation *> valTrack;    
    operandsToVisit.insert(val);
    while (operandsToVisit.size() > 0) {
      Value currOp = operandsToVisit.pop_back_val();
      visitedOperands.insert(currOp);
      Operation* defOp = currOp.getDefiningOp();
      // defOp is defined inside of loopOp
      auto operands = defOp -> getOperands(); 
      if (llvm::isa<memref::LoadOp, fir::LoadOp>(defOp)) {
        // Value of definition operation should not be a loadOp!
        return std::nullopt;
      }
      auto shouldTrackDefOp = false;
      for (auto operand : operands) {
        if (operand == IV) {
          // should not rely on IV
          return std::nullopt;
        }
        if (loopOp.isDefinedOutsideOfLoop(operand)) {
          continue;
        } 
        if (!visitedOperands.contains(operand)) {
          operandsToVisit.insert(operand);
          shouldTrackDefOp = true;
        }
      }
      if (shouldTrackDefOp) {
        valTrack.push_back(defOp);
      }
    }
    return valTrack;
  }

  // example: store %val to %mem[indices]
  // allowIV: allow value to be relied on IV, but not allow indices to rely on IV as it changes during each iteration.
  static LogicalResult collectDependencies(
    mlir::LoopLikeOpInterface loopOp,
    Value IV,
    Value val,
    llvm::DenseSet<Operation*>& opsToClone,
    bool allowIV
  ) {
    if (val == IV) {
      return allowIV? success(): failure();
    }

    llvm::SmallVector<Value> workList;
    workList.push_back(val);
    llvm::DenseSet<Value> visited;

    while (!workList.empty()) {
      Value currValue = workList.pop_back_val();
      if (visited.contains(currValue)) {
        continue;
      }
      if (currValue == IV) {
        if (allowIV) {
          continue;
        } else {
          return failure();
        }
      }

      if (loopOp.isDefinedOutsideOfLoop(currValue)) {continue;}

      Operation* defOp = currValue.getDefiningOp();
      if (!defOp) {
        return failure();
      }
      
      if (llvm::isa<memref::LoadOp, fir::LoadOp>(defOp)) {
        DEBUG_PRINT("Dependency chain relies on LoadOp!");
        return failure();
      }

      opsToClone.insert(defOp);
      for (Value operand: defOp->getOperands()) {
        workList.push_back(operand);
      }
    }
    return success();
  }

  static Operation* reconstructFinalIV(Operation* loopOp, PatternRewriter& rewriter) {
    auto loc = loopOp->getLoc();
    if (auto doLoopOp = llvm::dyn_cast<fir::DoLoopOp>(loopOp)) {
      // doloop start with index 1, and includes [lowerbound, upperbound]
      // final_iv = lb + ((ub - lb) / step) * step
      Value lb = doLoopOp.getLowerBound();
      Value ub = doLoopOp.getUpperBound();
      Value step = doLoopOp.getStep();

      Value diff = arith::SubIOp::create(rewriter, loc, ub.getType(), ub, lb, {});
      Value iters = arith::DivSIOp::create(rewriter, loc, diff.getType(), diff, step, {});
      Value offset = arith::MulIOp::create(rewriter, loc, iters.getType(), step, {});
      return arith::AddIOp::create(rewriter, loc, lb.getType(), lb, offset, {});
    } else if (auto affineForLoopOp = llvm::dyn_cast<affine::AffineForOp>(loopOp)) {
      // affineFor starts from 0, and it is [lb, ub)
      // final_iv = lb + ((ub - lb - 1) / step) * step
      //
      // affine.for iv = lb to ub step
      
      assert(affineForLoopOp.getLowerBound().getNumOperands() <= 1); // if constant bound, then this is zero
      assert(affineForLoopOp.getUpperBound().getNumOperands() <= 1);
      Value lb, ub;
      if (affineForLoopOp.hasConstantLowerBound()) {
        int64_t lbInt = affineForLoopOp.getConstantLowerBound();
        rewriter.setInsertionPoint(affineForLoopOp);
        lb = arith::ConstantIndexOp::create(rewriter, affineForLoopOp.getLoc(), lbInt);
      } else {
        lb = affineForLoopOp.getLowerBound().getOperand(0);
      }
      if (affineForLoopOp.hasConstantUpperBound()) {
        int64_t ubInt = affineForLoopOp.getConstantUpperBound(); 
        rewriter.setInsertionPoint(affineForLoopOp);
        ub = arith::ConstantIndexOp::create(rewriter, affineForLoopOp.getLoc(), ubInt);
      } else {
        ub = affineForLoopOp.getUpperBound().getOperand(0);
      }
      int64_t stepInt = affineForLoopOp.getStepAsInt();
      Value step = arith::ConstantIndexOp::create(rewriter, loc, stepInt);

      Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value realub = arith::SubIOp::create(rewriter, loc, ub.getType(), ub, c1, {});
      Value diff = arith::SubIOp::create(rewriter, loc, realub.getType(), realub, lb, {});
      Value iters = arith::DivSIOp::create(rewriter, loc, diff.getType(), diff, step, {});
      // static MulIOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::arith::IntegerOverflowFlags overflowFlags = ::mlir::arith::IntegerOverflowFlags::none);
      Value offset = arith::MulIOp::create(rewriter, loc, iters.getType(), iters, step, {});
      return arith::AddIOp::create(rewriter, loc, lb.getType(), lb, offset, {});
    }
    return nullptr;
  }

  static Operation* reconstructStore(
    Operation* writeOp, 
    Value val, 
    llvm::SmallVector<Value> indices, 
    PatternRewriter &rewriter
  ) {
    if (auto firStoreOp = llvm::dyn_cast<fir::StoreOp>(writeOp)) {
      fir::StoreOp::create(rewriter, writeOp->getLoc(), val, firStoreOp.getMemref());
    } else if (auto memStoreOp = llvm::dyn_cast<memref::StoreOp>(writeOp)){
      memref::StoreOp::create(rewriter, writeOp->getLoc(), val, memStoreOp.getMemref(), indices);
    } else {
      return nullptr;
    }
  }

  LogicalResult matchAndRewrite(WriteOpTy writeOp,
                                PatternRewriter &rewriter) const final {
    mlir::LoopLikeOpInterface loopOp = writeOp->template getParentOfType<LoopLikeOpInterface>();
    if (!loopOp) {
      return failure();
    }

    Value mem = writeOp.getMemref();
    Value valueToStore = writeOp.getValue();

    bool hasOtherMemAccess = false;
    bool hasBreak = false;
    loopOp->walk([&](Operation* op){
      if (isOperationPossiblelyReadFromAddr(op, mem)) {
        hasOtherMemAccess = true;
        WalkResult::interrupt();
      } 
      if (isOperationPossiblelyWriteToAddr(op, mem)) {
        if (op != writeOp) {
          hasOtherMemAccess = true;
          WalkResult::interrupt();
        }
      }
      if (llvm::isa<func::ReturnOp>(op)) {
        hasBreak = true;
        WalkResult::interrupt(); 
      }
      WalkResult::advance();
    });
    if (hasOtherMemAccess || hasBreak) {
      return failure();
    }

    // reconstruct value to store: value should only be made of constants or IV
    Value IV = getIV(loopOp);
    if (!IV) {
      return failure();
    }

    llvm::DenseSet<Operation*> opsToClone;
    if (failed(collectDependencies(loopOp, IV, valueToStore, opsToClone, true))) {
      return failure();
    }
    
    // If memref, we should also reconstruct the indices 
    if constexpr (std::is_same_v<WriteOpTy, memref::StoreOp>) {
      auto indices = writeOp.getIndices();
      for (const auto& idx: indices) {
        if (failed(collectDependencies(loopOp, IV, idx, opsToClone, false))) {
          return failure();
        }    
      }
    }
  
    // reconstruct IV
    Operation* finalIVOp = reconstructFinalIV(loopOp, rewriter);
    rewriter.setInsertionPointAfter(loopOp);
    assert(finalIVOp->getNumResults() == 1);

    // copy the value track and indices track
    IRMapping mapping;
    mapping.map(IV, finalIVOp->getResult(0));
    loopOp.walk([&](Operation* op){
      if (opsToClone.contains(op)) {
        rewriter.clone(*op, mapping);
      }
    });
    Value resVal = mapping.lookupOrDefault(valueToStore);

    // reconstruct the write operation
    if constexpr (std::is_same_v<WriteOpTy, fir::StoreOp>) {
      fir::StoreOp::create(rewriter, writeOp->getLoc(), resVal, mem);
    } else if constexpr (std::is_same_v<WriteOpTy, memref::StoreOp>) {
      llvm::SmallVector<Value> resIndices;
      for (Value idx: writeOp.getIndices()) {
        resIndices.push_back(mapping.lookupOrDefault(idx));
      }
      memref::StoreOp::create(rewriter, writeOp->getLoc(), resVal, mem, resIndices);
    } else {
      llvm::errs() << "Unexpected write type!\n";
      return failure();
    }    
    
    // erase original store
    rewriter.eraseOp(writeOp);
    return success();
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
    patterns.add<HoistWriteOpFromLoop<fir::StoreOp>>(ctx);
    patterns.add<HoistWriteOpFromLoop<memref::StoreOp>>(ctx);
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
