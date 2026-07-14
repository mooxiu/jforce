#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
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
  int64_t lb; // start, zero based
  int64_t ub; // limit
  int64_t step; // stride
};

// For example: %slice = %v[1:5:1, 2:5:2];
// Then we can represent slice as (%v, {(1, 5, 1), (2, 5, 2)});
struct RefValue {
  Value root;
  llvm::SmallVector<Triplet> triplets;

  llvm::SmallVector<llvm::SmallVector<int64_t>, 3> getZipTriplets() {
    llvm::SmallVector<llvm::SmallVector<int64_t>, 3> res;
    if (this->triplets.empty()) {
      return res;
    }
    llvm::SmallVector<int64_t> starts;
    llvm::SmallVector<int64_t> limits;
    llvm::SmallVector<int64_t> strides;
    for (const auto& triplet: this->triplets) {
      starts.push_back(triplet.lb);
      limits.push_back(triplet.ub);
      strides.push_back(triplet.step);
    }
    res.push_back(starts);
    res.push_back(limits);
    res.push_back(strides);
    return res;
  };
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

  Value slicing(
    OpBuilder& opBuilder,
    Value hloVal, 
    const llvm::SmallVector<llvm::SmallVector<int64_t>, 3>& zipTriplets 
  ) {
    if (zipTriplets.empty() || zipTriplets[0].empty()) {
      return hloVal;
    }
    auto toAttr = [](MLIRContext* ctx, llvm::SmallVector<int64_t> intList) -> ::mlir::DenseI64ArrayAttr {
      llvm::SmallVector<long> longList = llvm::to_vector(llvm::map_range(intList, [](int64_t ele){return long(ele);}));
      return DenseI64ArrayAttr::get(ctx, longList);
    };
    MLIRContext* ctx = opBuilder.getContext();
    auto sliceOp = stablehlo::SliceOp::create(
      opBuilder,
      hloVal.getLoc(),
      hloVal,
      /*start_indices=*/ toAttr(ctx, zipTriplets[0]),
      toAttr(ctx, zipTriplets[1]),
      toAttr(ctx, zipTriplets[2])
    );
    return sliceOp.getResult();
  } 

  Value getReferenceValue(
    OpBuilder& opBuilder,
    Value ref
  ) {
    assert(this->referenceMap.contains(ref) && "This is not a ref!");
    RefValue refVal = referenceMap.at(ref); 
    assert(memoryMap.contains(refVal.root));
    auto rootVal = memoryMap.at(refVal.root);
    if (refVal.triplets.empty()) {
      return rootVal;
    } else {
      return slicing(opBuilder, rootVal, refVal.getZipTriplets());
    }
  }

  Value getValue(OpBuilder& opBuilder, Value memOrVal) {
    if (this->valueMap.contains(memOrVal)) {
      return this->valueMap.at(memOrVal);
    } else {
      return this->getReferenceValue(opBuilder, memOrVal);
    }
    memOrVal.dump();
    llvm_unreachable("Value is neither an SSA value nor a reference");
  }
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
    state.referenceMap[oldArg] = {oldArg, {}};
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
  opBuilder.setInsertionPointToEnd(&(stableHLOFuncOp.front()));

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

// Fortran index can start from minus value, we should extract the information from sliceShiftMap.
static llvm::SmallVector<int64_t> getLowerBounds(
  hlfir::DesignateOp designateOp,
  const llvm::DenseMap<Value, llvm::SmallVector<int64_t>>& sliceShiftMap
) {
  size_t rank = designateOp.getIsTriplet().size();
  llvm::SmallVector<int64_t> lowerBounds(rank, 1); // initiated as 1
  auto declareOp = designateOp.getMemref().getDefiningOp<hlfir::DeclareOp>();
  if (!declareOp) return lowerBounds;
  Value shape = declareOp.getShape();
  if (!shape) return lowerBounds;
  if (auto shapeShiftOp = shape.getDefiningOp<fir::ShapeShiftOp>()) {
    auto it = sliceShiftMap.find(shapeShiftOp.getResult());
    assert(it != sliceShiftMap.end() && "Static shape_shift lower bounds must be recorded");
    assert(it->second.size() == rank && "Lower-bound count must match designate rank");
    lowerBounds = it->second;
  } else {
    assert(shape.getDefiningOp<fir::ShapeOp>() && "Expected fir.shape or fir.shape_shift");
  }
  return lowerBounds;
}

static llvm::SmallVector<Triplet> extractStaticTriplets(
  hlfir::DesignateOp designateOp,
  llvm::ArrayRef<int64_t> lowerBounds
) {
  auto indices = designateOp.getIndices();
  auto isTriplet = designateOp.getIsTriplet();

  assert(!isTriplet.empty());
  assert(llvm::all_of(isTriplet, [](bool flag) { return flag; }) && "Only pure-triplet designates are supported");
  assert(indices.size() == isTriplet.size() * 3);
  assert(lowerBounds.size() == isTriplet.size());

  auto getConstantInt = [](Value val) {
    IntegerAttr attr;
    if (matchPattern(val, m_Constant(&attr))) {
      return attr.getInt();
    };
    llvm_unreachable("Should be constant");
  };

  llvm::SmallVector<Triplet> result;
  result.reserve(isTriplet.size());
  auto it = indices.begin();
  for (auto [dim, flag] : llvm::enumerate(isTriplet)) {
    assert(flag);
    int64_t fortranLower = getConstantInt(*it++);
    int64_t fortranUpper = getConstantInt(*it++);
    int64_t stride = getConstantInt(*it++);
    assert(stride > 0 && "Only positive static strides are supported");
    int64_t arrayLowerBound = lowerBounds[dim];
    result.push_back({
        .lb = fortranLower - arrayLowerBound,
        .ub = fortranUpper - arrayLowerBound + 1,
        .step = stride,
    });
  }
  assert(it == indices.end());
  return result;
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
      assert(state.referenceMap.contains(memRef));
      assert(declareOp.getNumResults() == 2);
      for (int i = 0; i < declareOp.getNumResults(); i++ ) {
        auto result = declareOp.getResult(i);
        state.referenceMap[result] = state.referenceMap.at(memRef);
      }
    })
    .Case([&](hlfir::AssignOp assignOp){
      auto assignTo = assignOp.getLhs();
      auto assignFrom = assignOp.getRhs();
      assert(state.referenceMap.contains(assignTo));
      assert(state.referenceMap.contains(assignFrom) || state.valueMap.contains(assignFrom));

      Value assignFromVal = state.getValue(opBuilder, assignFrom);
      
      
      // assignTo can be a slice of a memory, so we have to only update some of the memory.
      auto assignToRefVal = state.referenceMap.at(assignTo);
      if (assignToRefVal.triplets.empty()) {
        state.memoryMap[assignToRefVal.root] = assignFromVal;
      } else {
        auto LHSZipTriplets = assignToRefVal.getZipTriplets(); 
        auto isDenseUpdate = llvm::all_of(LHSZipTriplets[2], [](int64_t stride){return stride == 1;});
        Value updatedVal; 
        if (isDenseUpdate) {
          // Update with dynamic_slice_update
          llvm::SmallVector<Value> startIndices;
          llvm::for_each(LHSZipTriplets[0], [&](int64_t idx){
            auto type = RankedTensorType::get({}, opBuilder.getI64Type());
            auto attr = DenseIntElementsAttr::get(type, {llvm::APInt(64, idx, /*isSigned=*/true)});
            auto constOp = stablehlo::ConstantOp::create(
              opBuilder,
              assignOp.getLoc(),
              attr
            );
            startIndices.push_back(constOp.getResult());
          });

          auto assignedToRootVal = state.memoryMap.at(assignToRefVal.root);
          auto updateOp = stablehlo::DynamicUpdateSliceOp::create(
            opBuilder, 
            assignOp.getLoc(),
            /*result=*/ assignedToRootVal.getType(),
            assignedToRootVal,
            assignFromVal,
            startIndices
          );
          updatedVal = updateOp.getResult();
        } else {
          // TODO: Update with scatter
          llvm_unreachable("Implement me!"); 
        }
        state.memoryMap[assignToRefVal.root] = updatedVal;
      }
    })
    .Case([&](fir::LoadOp loadOp){
      auto resType = loadOp.getResult().getType(); 
      if (fir::isa_ref_type(resType) || fir::isa_box_type(resType)) {
        llvm_unreachable("Only support LoadOp result is a plain value for now!");
      }
      auto loadedVal = state.getReferenceValue(opBuilder, loadOp.getMemref());
      state.valueMap[loadOp.getResult()] = loadedVal;
    })
    .Case([&](hlfir::DesignateOp designateOp) {
      auto lowerBounds = getLowerBounds(designateOp, sliceShiftMap);
      auto triplets = extractStaticTriplets(designateOp, lowerBounds);
      std::reverse(triplets.begin(), triplets.end()); // Fortran is column based, so should reverse
      assert(state.referenceMap.contains(designateOp.getMemref()));
      auto memRefVal = state.referenceMap.at(designateOp.getMemref());
      assert(memRefVal.triplets.empty()); // TODO: it can be a slice of a slice, so actually we should cover the composed triplets!
      state.referenceMap[designateOp.getResult()] = RefValue{
        .root = memRefVal.root,
        .triplets = std::move(triplets),
      };
    })
    .Case([&](fir::AllocaOp allocaOp){
      state.referenceMap[allocaOp.getResult()] = {allocaOp.getResult(), {}};
    })
    // INFO: built-in array operations
    .Case([&](hlfir::TransposeOp transposeOp){
      auto opTy = toCorrespondingTensorTy(transposeOp.getOperand().getType());
      auto resTy = RankedTensorType::get(
        llvm::SmallVector<int64_t>{opTy.getShape()[1], opTy.getShape()[0]},
        opTy.getElementType());
      auto stablehloTransposeOp = stablehlo::TransposeOp::create(
        opBuilder, op->getLoc(), resTy, state.getValue(opBuilder, transposeOp.getOperand()),
        llvm::SmallVector<int64_t>{1, 0});
      state.valueMap[transposeOp.getResult()] = stablehloTransposeOp.getResult();
    })
    .Case([&](hlfir::MatmulOp mmOp){
      // %36 = hlfir.matmul %33#0 %35#0 {fastmath = #arith.fastmath<contract>}
      // : (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>) ->
      // !hlfir.expr<?x?xf64>
      auto op0 = state.getValue(opBuilder, mmOp.getOperand(0));
      auto op1 = state.getValue(opBuilder, mmOp.getOperand(1));
      auto op0Ty = toCorrespondingTensorTy(op0.getType());
      auto op1Ty = toCorrespondingTensorTy(op1.getType());
      auto resTy = RankedTensorType::get(
          llvm::SmallVector<int64_t>{op1Ty.getShape()[0],
                                     op0Ty.getShape()[1]},
          op0Ty.getElementType());
      mlir::ArrayAttr config = {};
      mlir::stablehlo::DotAlgorithmAttr algo = {};
      auto dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
          op->getContext(), SmallVector<int64_t>{}, SmallVector<int64_t>{},
          SmallVector<int64_t>{1}, SmallVector<int64_t>{0});
      auto stablehloDotGeneralOp = stablehlo::DotGeneralOp::create(
          opBuilder, op->getLoc(), resTy, op1, op0, dims, config, algo);
      state.valueMap[mmOp.getResult()] = stablehloDotGeneralOp.getResult();
    })
    .Case<fir::ShapeOp, fir::ShapeShiftOp, hlfir::DestroyOp,
      func::ReturnOp, omp::WorkdistributeOp, omp::TeamsOp, omp::TerminatorOp>([](auto) {
        // No runtime tensor semantics.
    })
    .Default([&](auto unsupportedOp){
      unsupportedOp->dump();
      llvm_unreachable("Unsupported operation in TranslatePass");
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
