#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include "../support/profiler.h"
#include "../support/utilities.h"
#include "transform/Utils.h"

using namespace mlir;

#define JIT_LITERAL_VAL_ATTR_NAME "jit.literal_val"
#define JIT_SLICE_SHIFT_ATTR_NAME "jit.slice_shift"

namespace {

struct Triplet {
  int lb;
  int ub;
  int step;
};

// For example: %slice = %v[1:5:1, 2:5:2];
// Then we can represent slice as (%v, {(1, 5, 1), (2, 5, 2)});
struct RefValue {
  Value root;
  llvm::SmallVector<Triplet> triplets;
};

struct TranslationState {
  // Fortran's IR value to StableHLO IR value.
  llvm::DenseMap<Value, Value> valueMap;
  // Record slice value -> ref value representation 
  llvm::DenseMap<Value, RefValue> referenceMap;
  // Fortran IR's memory SSA -> tensor 
  llvm::DenseMap<Value, Value> memoryMap;
  // Arguments of Fortran'IR
  // When return, for each of them find the corresponding tensor through memoryMap, and return.
  llvm::SmallVector<Value> argList;
};

///  Example of source type:
///  "!fir.ref<!fir.array<10xf32>>": convert to "tensor<10xf32>"
///  "!fir.ref<!fir.array<10x20xf32>>": convert to "tensor<20x10xf32>", notice
///  it is reversed
///  "!fir.ref<f32>": convert to "tensor<f32>"
static RankedTensorType toCorrespondingTensorTy(mlir::Type srcTy) {
  // If it's already a tensor type, then no need to convert
  if (llvm::isa<RankedTensorType>(srcTy)) {
    return llvm::dyn_cast<RankedTensorType>(srcTy);
  }

  return llvm::TypeSwitch<mlir::Type, RankedTensorType>(srcTy)
      .Case<hlfir::ExprType>([](hlfir::ExprType expTy) {
        auto shape = llvm::to_vector(expTy.getShape());
        std::reverse(shape.begin(), shape.end());
        return RankedTensorType::get(shape, expTy.getEleTy());
      })
      .Case<fir::BoxType>([](fir::BoxType bTy) {
        return toCorrespondingTensorTy(bTy.getEleTy());
      })
      .Case<fir::ReferenceType>([](fir::ReferenceType refTy) {
        return toCorrespondingTensorTy(refTy.getEleTy());
      })
      .Case<fir::SequenceType>([](fir::SequenceType seqTy) {
        auto shape = llvm::to_vector(seqTy.getShape());
        std::reverse(shape.begin(), shape.end());
        return RankedTensorType::get(shape, seqTy.getEleTy());
      })
      .Case<mlir::MemRefType>([](MemRefType memTy){
        auto shape = llvm::to_vector(memTy.getShape());
        std::reverse(shape.begin(), shape.end());
        return RankedTensorType::get(shape, memTy.getElementType());
      })
      .Default([&](auto scTy) {
        // Suppose this is a scalar type
        return RankedTensorType::get({}, scTy);
      });
}


static func::FuncOp createFunction(
  OpBuilder& opBuilder,
  TranslationState& state, 
  func::FuncOp firFunc
) {
  OpBuilder::InsertionGuard guard(opBuilder);
  opBuilder.setInsertionPointAfter(firFunc);

  llvm::SmallVector<Type> argTypes;
  llvm::for_each(firFunc.getArgumentTypes(), [&](const Type argType){
    argTypes.push_back(toCorrespondingTensorTy(argType));
    return;
  });
  auto funcType = FunctionType::get(firFunc.getContext(), argTypes, argTypes);
  auto stableHLOFunc = func::FuncOp::create(opBuilder, firFunc.getLoc(), "main", funcType, {}, {});
  stableHLOFunc.addEntryBlock();

  for (int i = 0; i < firFunc.getNumArguments(); i++) {
    auto oldArg = firFunc.getArgument(i);
    auto newArg = stableHLOFunc.getArgument(i);
    state.argList.push_back(oldArg);
    state.memoryMap[oldArg] = newArg;
    auto argAttr = firFunc.getArgAttrOfType<IntegerAttr>(i, JIT_LITERAL_VAL_ATTR_NAME);
    if (argAttr) {
      stableHLOFunc.setArgAttr(i, JIT_LITERAL_VAL_ATTR_NAME, argAttr);
    }
  }
  return stableHLOFunc;
}

static void terminateFunction(
  OpBuilder& opBuilder, 
  const TranslationState& state, 
  func::FuncOp stableHLOFuncOp
) {
  OpBuilder::InsertionGuard guard(opBuilder);
  opBuilder.setInsertionPointToEnd(&(stableHLOFuncOp.back()));

  llvm::SmallVector<Value> resultList;
  for (auto arg: state.argList) {
    assert(state.memoryMap.contains(arg));
    resultList.push_back(state.memoryMap.at(arg));
  }
  func::ReturnOp::create(opBuilder, stableHLOFuncOp.getLoc(), resultList);
  return;
};


static void translateElemental(
  OpBuilder& opBuilder, 
  TranslationState& state, 
  func::FuncOp newFunc, 
  hlfir::ElementalOp elementalOp,
  const llvm::DenseMap<Value, llvm::SmallVector<int64_t>>& sliceShiftMap
) {
  
}

static void translateOperation(
  OpBuilder& opBuilder, 
  TranslationState& state, 
  func::FuncOp newFunc, 
  Operation* op,
  const llvm::DenseMap<Value, llvm::SmallVector<int64_t>>& sliceShiftMap
) {
  OpBuilder::InsertionGuard guard(opBuilder);
  opBuilder.setInsertionPointToEnd(&newFunc.front());

  llvm::TypeSwitch<Operation*>(op)
    .Case([&](arith::ConstantOp constOp){
      stablehlo::ConstantOp stablehloConstOp;
      if (constOp.getType().isIndex()) {
        auto indexAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValueAttr());
        auto intAttr = IntegerAttr::get(IntegerType::get(opBuilder.getContext(), 64), indexAttr.getValue());
        stablehloConstOp = stablehlo::ConstantOp::create(opBuilder, newFunc.getLoc(), intAttr);
      } else {
        stablehloConstOp = stablehlo::ConstantOp::create(opBuilder, newFunc.getLoc(), constOp.getValueAttr());
      }
      state.valueMap[constOp.getResult()] = stablehloConstOp.getResult();
    })
    .Case([&](hlfir::DeclareOp declareOp){
      auto memRef = declareOp.getMemref();
      assert(declareOp.getNumResults() == 2);
      for (int i = 0; i < declareOp.getNumResults(); i++ ) {
        auto result = declareOp.getResult(i);
        RefValue refVal{memRef, {}};
        state.referenceMap[result] = refVal;
      }
    })
    .Case([&](hlfir::AssignOp assignOp){
      auto assignedMem = assignOp.getLhs();
      auto assignVal = assignOp.getRhs();
      assert(state.memoryMap.contains(assignedMem));
      assert(state.valueMap.contains(assignVal));
      state.memoryMap[assignedMem] = state.valueMap[assignVal]; 
    })
    .Default([&](auto){
      DEBUG_PRINT("Skipped During Translation: ");
      DEBUG_PRINT_OP(op);
    });
  return;
}

static void translateRegion(
  OpBuilder& opBuilder, 
  TranslationState& state, 
  func::FuncOp newFunc, 
  Region& region,
  const llvm::DenseMap<Value, llvm::SmallVector<int64_t>>& sliceShiftMap
) {
  for (Operation &op: region.front()) {
    if (auto elementalOp = llvm::dyn_cast<hlfir::ElementalOp>(op)) {
      translateElemental(opBuilder, state, newFunc, elementalOp, sliceShiftMap);
      continue;
    }
    if (!op.getRegions().empty()) {
      for (auto& nestRegion: op.getRegions()) {
        translateRegion(opBuilder, state, newFunc, nestRegion, sliceShiftMap);
      }
    }
    translateOperation(opBuilder, state, newFunc, &op, sliceShiftMap);
  }
}


struct TranslatePass
    : public PassWrapper<TranslatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TranslatePass)

  llvm::DenseMap<Value, llvm::SmallVector<int64_t>> extractSliceShifts(func::FuncOp funcOp) {
    llvm::DenseMap<Value, llvm::SmallVector<int64_t>> sliceShiftMap;
    funcOp.walk([&](fir::ShapeShiftOp ssOp){
      auto shifts = ssOp->getAttrOfType<DenseI64ArrayAttr>(JIT_SLICE_SHIFT_ATTR_NAME);
      if (shifts && !shifts.empty()) {
        llvm::SmallVector<int64_t> shiftsVec(shifts.asArrayRef());
        sliceShiftMap[ssOp.getResult()] = shiftsVec;
      }
    });
    return sliceShiftMap;
  };

  void getDependentDialects(mlir::DialectRegistry & registry) const override {
    registry.insert<stablehlo::StablehloDialect>();
  }

  StringRef getArgument() const override { 
    return "jforce-translatev2"; 
  }

  void runOnOperation() override {
    auto moduleOp = getOperation(); 
    auto context = moduleOp.getContext();
    OpBuilder opBuilder(context);


    // Add funcs to transform into list
    llvm::SmallVector<func::FuncOp>  funcsToReplace;
    moduleOp.walk([&](func::FuncOp fOp){
      funcsToReplace.push_back(fOp);
    });

    for (func::FuncOp oldFOp: funcsToReplace) {
      TranslationState state;
      auto sliceShiftMap = extractSliceShifts(oldFOp);
      auto stableHLOFuncOp = createFunction(opBuilder, state, oldFOp);
      translateRegion(opBuilder, state, stableHLOFuncOp, oldFOp.getRegion(), sliceShiftMap);
      terminateFunction(opBuilder, state, stableHLOFuncOp);
      oldFOp.erase();
    }
  }
};
} // namespace

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createTranslatePass() {
    return std::make_unique<TranslatePass>();
  }

  void registerTranslatePass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createTranslatePass();});
  };
}
