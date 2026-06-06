#include "MemUtils.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "support/utilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
namespace {

// TODO: need to do alias analysis to make this more strict
template<typename IfTy, typename LoadTy>
struct HoistLoadOps: OpRewritePattern<IfTy> {
  using OpRewritePattern<IfTy>::OpRewritePattern;

  // FIXME: finish this
  static std::optional<Value> getWrittenMemory(const Operation& op) {
    return std::nullopt;
  }

  static void getWrittenMemories(Block& block, llvm::DenseSet<Value>& writtenMemSet) {
    for(const auto& op : block.getOperations()) {
      // TODO: also need to check other blocks
      if (auto nestedIfOp = llvm::dyn_cast<fir::IfOp>(op)) {
        getWrittenMemories(nestedIfOp.getThenRegion().front(), writtenMemSet);
        if (!nestedIfOp.getElseRegion().empty()) {
          getWrittenMemories(nestedIfOp.getElseRegion().front(), writtenMemSet);
        }
      } else {
        auto memOptional = getWrittenMemory(op);
        if (memOptional.has_value()) {
          writtenMemSet.insert(memOptional.value());
        }
      }
    }
  }

  LogicalResult matchAndRewrite(IfTy ifOp,
                                PatternRewriter &rewriter) const final {
    
    // if hoist something, return true; else return false.
    auto getHoistableLoadOps = [](Block& block) -> llvm::SmallVector<Operation*> {
      llvm::SmallVector<Operation*> hoistableLoadOps;
      llvm::DenseSet<Value> possiblyWrittenMems;
      llvm::for_each(block.getOperations(), [&](Operation& op){
        if (auto loadOp = llvm::dyn_cast<LoadTy>(&op)) {
          if (!possiblyWrittenMems.contains(loadOp.getMemref())) {
            hoistableLoadOps.push_back(&op);
          }
          return;
        }
        if (auto nestedIfOp = llvm::dyn_cast<IfTy>(&op)) {
          getWrittenMemories(nestedIfOp.getThenRegion().front(), possiblyWrittenMems);
          if (!nestedIfOp.getElseRegion().empty()) {
            getWrittenMemories(nestedIfOp.getElseRegion().front(), possiblyWrittenMems);
          }
        } else {
          auto memOpt = getWrittenMemory(op);
          if (memOpt.has_value()) {
            possiblyWrittenMems.insert(memOpt.value());
          }
        }
      });
      return hoistableLoadOps;
    };

    rewriter.setInsertionPoint(ifOp);
    auto& thenBlock = ifOp.getThenRegion().front();
    bool changed = false;
    for (Operation* op: getHoistableLoadOps(thenBlock)) {
      rewriter.moveOpBefore(op, ifOp);
      if (!changed) {
        changed = true;
      }
    }

    if (!ifOp.getElseRegion().empty()) {
      auto& elseBlock = ifOp.getElseRegion().front();
      for (Operation* op: getHoistableLoadOps(elseBlock)) {
        rewriter.moveOpBefore(op, ifOp);
        if (!changed) {
          changed = true;
        }
      }
    }
    return changed? success(): failure();
  }
};

struct SinkStoreOps: OpRewritePattern<fir::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  static void getSinkableOperations(Block& block, llvm::DenseMap<Value, Operation*>& sinkableStoreOps) {
    for (auto& op: block.getOperations()) {
      Operation* opPtr = &op;
      if (auto assignOp = llvm::dyn_cast<hlfir::AssignOp>(&op)) {
        auto mem = assignOp.getLhs();
        auto val = assignOp.getRhs();
        if (val.getDefiningOp()->getBlock() != &block) {
          sinkableStoreOps[mem] = assignOp;
        }
        continue;
      }
      
      if (auto nestedIfOp = llvm::dyn_cast<fir::IfOp>(op)) {
        auto& nestedThenBlock = nestedIfOp.getThenRegion().front();
        getSinkableOperations(nestedThenBlock, sinkableStoreOps);
        if (!nestedIfOp.getElseRegion().empty()) {
          auto& nestedElseBlock = nestedIfOp.getElseRegion().front();
          getSinkableOperations(nestedElseBlock, sinkableStoreOps);
        }
        continue;
      }

      for (const auto& entry: sinkableStoreOps) {
        if (isOperationPossiblelyReadFromAddr(&op, entry.getFirst())) {
          sinkableStoreOps.erase(entry.getFirst());
        }
      }
    }
  }

  // TODO: should use MemOpsFoldingPass to run first.
  static llvm::DenseMap<Value, std::pair<Operation*, Operation*>> getUnionMap(
    llvm::DenseMap<Value, Operation*>& thenBlockSinkableOperations, 
    llvm::DenseMap<Value, Operation*>& elseBlockSinkableOperations
  ) {
    llvm::DenseMap<Value, std::pair<Operation*, Operation*>> unionMap;
    assert(!thenBlockSinkableOperations.empty());
    assert(!elseBlockSinkableOperations.empty());
    for (auto& entry: thenBlockSinkableOperations) {
      auto* thenOp = entry.getSecond();
      auto memref = entry.getFirst();
      auto it = elseBlockSinkableOperations.find(memref);
      if (it != elseBlockSinkableOperations.end()) {
        auto* elseOp = it->getSecond();
        unionMap[it->getFirst()] = std::pair<Operation*, Operation*>(thenOp, elseOp);
      }
    }
    return unionMap;
  }

  static bool sinkOperations(
    fir::IfOp ifOp,
    llvm::DenseMap<Value, Operation*>& thenBlockSinkableOperations, 
    llvm::DenseMap<Value, Operation*>& elseBlockSinkableOperations, 
    PatternRewriter& rewritter
  ) {
    if (thenBlockSinkableOperations.empty() || elseBlockSinkableOperations.empty()) {
      return false;
    }
    auto unionMap = getUnionMap(thenBlockSinkableOperations, elseBlockSinkableOperations);
    if (unionMap.empty()) {
      return false;
    }

    for (auto& entry: unionMap) {
      auto* op1 = entry.getSecond().first;
      auto* op2 = entry.getSecond().second;
      assert(llvm::isa<hlfir::AssignOp>(op1));
      assert(llvm::isa<hlfir::AssignOp>(op2));
      auto assignOp1 = llvm::dyn_cast<hlfir::AssignOp>(op1);
      auto assignOp2 = llvm::dyn_cast<hlfir::AssignOp>(op2);

      rewritter.setInsertionPointAfter(ifOp);
      // copy definition chain

      // generating select and assign
      auto valType = assignOp1.getRhs().getType();
      auto memRef = entry.getFirst();
      auto selectOp = arith::SelectOp::create(
        rewritter, 
        ifOp.getLoc(), 
        valType,
        ifOp.getCondition(),
        assignOp1.getRhs(),
        assignOp2.getRhs()
      );
      hlfir::AssignOp::create(
        rewritter,
        ifOp.getLoc(),
        selectOp.getResult(),
        memRef
      );
      rewritter.eraseOp(assignOp1);
      rewritter.eraseOp(assignOp2);
    }
    return true;
  }

  LogicalResult matchAndRewrite(fir::IfOp ifOp,
                                PatternRewriter &rewriter) const final {
    llvm::DenseMap<Value, Operation*> thenStoreOps; 
    llvm::DenseMap<Value, Operation*> elseStoreOps; 
    auto& thenBlock = ifOp.getThenRegion().front(); 
    getSinkableOperations(thenBlock, thenStoreOps);
    if (!ifOp.getElseRegion().empty()) {
      auto& elseBlock = ifOp.getElseRegion().front(); 
      getSinkableOperations(elseBlock, elseStoreOps);
    }

    return sinkOperations(ifOp, thenStoreOps, elseStoreOps, rewriter)? success(): failure();
  }
};

template<typename IfTy>
struct HoistIfInvariantArithOps: OpRewritePattern<IfTy> {
  using OpRewritePattern<IfTy>::OpRewritePattern;

  static llvm::SmallVector<Operation*> getHoistableOpsInBlock(Block& block, IfTy ifOp) {
    llvm::SmallVector<Operation*> opsToHoist;
    for (Operation& op: block.getOperations()) {
      mlir::Dialect* dialect = op.getDialect();
      if (!llvm::isa<mlir::arith::ArithDialect, mlir::math::MathDialect>(dialect)) return {};

      bool allDefinedOut = true;
      for (const auto& operand: op.getOperands()) {
        if (ifOp->isAncestor(operand.getDefiningOp())) {
          allDefinedOut = false;
          break;
        }
      }
      if (allDefinedOut) {
        Operation* opPtr = &op;
        opsToHoist.push_back(opPtr);
      }
    }
    return opsToHoist;
  }

  LogicalResult matchAndRewrite(IfTy ifOp,
                                PatternRewriter &rewriter) const final {

    auto hoistOp = [&](Block& block) -> bool {
      auto ops = getHoistableOpsInBlock(block, ifOp);
      DEBUG_PRINT("The size of ops is: " + std::to_string(ops.size()));
      if (!ops.empty()) {
        llvm::for_each(ops, [&](Operation* op){rewriter.moveOpBefore(op, ifOp);});   
        return true;
      }
      return false;
    };

    auto moved = false;
    auto& thenBlock = ifOp.getThenRegion().front();
    moved = hoistOp(thenBlock) || moved;
    
    if (!ifOp.getElseRegion().empty()) {
      auto& elseBlock = ifOp.getElseRegion().front();
      moved = hoistOp(elseBlock) || moved;
    }
    
    return moved? success(): failure();
  }
};


struct IfConversionPass
    : public mlir::PassWrapper<IfConversionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hlfir::hlfirDialect, fir::FIROpsDialect>();
    return;
  }

  StringRef getArgument() const override { return "jforce-if-conversion"; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto ctx = getOperation()->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<HoistLoadOps<fir::IfOp, fir::LoadOp>>(ctx);
    patterns.add<HoistLoadOps<scf::IfOp, memref::LoadOp>>(ctx);
    patterns.add<SinkStoreOps>(ctx);
    patterns.add<HoistIfInvariantArithOps<fir::IfOp>>(ctx);
    patterns.add<HoistIfInvariantArithOps<scf::IfOp>>(ctx);
    GreedyRewriteConfig config;

    config.enableFolding();
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      funcOp.dump();
      signalPassFailure();
      return;
    }
    return;
  }
};
} // namespace

namespace xla_jit {
std::unique_ptr<mlir::Pass> createIfConversionPass() {
  return std::make_unique<IfConversionPass>();
}

void registerIfConversionPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createIfConversionPass(); });
};
} // namespace xla_jit
