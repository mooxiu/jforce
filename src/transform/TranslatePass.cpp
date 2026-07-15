#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <utility>
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

  // Get the corresponding tensor value in stablehlo function.
  Value getTensorValue(OpBuilder& opBuilder, Value memOrVal) {
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

// TODO: only support 0 rank broadcasting
static std::pair<Value, Value> alignShapes(
  OpBuilder& opBuilder,
  Location loc,
  Value tensorVal0, 
  Value tensorVal1
) {
  assert(llvm::isa<RankedTensorType>(tensorVal0.getType()));
  assert(llvm::isa<RankedTensorType>(tensorVal1.getType()));
  auto typeInfo0 = inspectTypeInfo(tensorVal0.getType());
  auto typeInfo1 = inspectTypeInfo(tensorVal1.getType());
  if (typeInfo0.rank == typeInfo1.rank) {
    assert(typeInfo0.shape == typeInfo1.shape);
    return std::pair<Value, Value>(tensorVal0, tensorVal1);
  }
  if (typeInfo0.rank < typeInfo1.rank) {
    auto broadcastOp = stablehlo::BroadcastInDimOp::create(
      opBuilder, 
      loc,
      tensorVal1.getType(),
      tensorVal0,
      opBuilder.getDenseI64ArrayAttr({})
    );
    return std::make_pair(broadcastOp.getResult(), tensorVal1);
  } else {
    // hlo0 has larger rank
    auto broadcastOp = stablehlo::BroadcastInDimOp::create(
      opBuilder, 
      loc,
      tensorVal0.getType(),
      tensorVal1,
      opBuilder.getDenseI64ArrayAttr({})
    );
    return std::make_pair(tensorVal0, broadcastOp.getResult());
  }
}


static Value handleBinaryArithOp(
  OpBuilder& opBuilder, 
  TranslationState& state, 
  Operation* op, 
  const llvm::SmallVector<Value>& args
) {
  Location loc = op->getLoc();
  auto opPair = alignShapes(opBuilder, loc, args[0], args[1]);
  auto hloO0 = opPair.first;
  auto hloO1 = opPair.second;
  Operation* createdHLOOp = llvm::TypeSwitch<Operation*, Operation*>(op)
    .Case<arith::AddFOp, arith::AddIOp>([&](auto){
      auto resTy = hloO0.getType();
      return stablehlo::AddOp::create(opBuilder, loc, resTy, hloO0, hloO1);
    })
    .Case<arith::SubFOp, arith::SubIOp>([&](auto){
      auto resTy = hloO0.getType();
      return stablehlo::SubtractOp::create(opBuilder, loc, resTy, hloO0, hloO1);
    })
    .Case<arith::MulFOp, arith::MulIOp>([&](auto){
      auto resTy = hloO0.getType();
      return stablehlo::MulOp::create(opBuilder, loc, resTy, hloO0, hloO1);
    })
    .Case<arith::DivFOp, arith::DivSIOp>([&](auto){
      auto resTy = hloO0.getType();
      return stablehlo::DivOp::create(opBuilder, loc, resTy, hloO0, hloO1);
    })
    .Case<arith::CmpFOp>([&](arith::CmpFOp cmpOp){
      stablehlo::ComparisonDirection direction;
      switch (cmpOp.getPredicate()) {
        case arith::CmpFPredicate::OEQ:
        case arith::CmpFPredicate::UEQ:
          direction = stablehlo::ComparisonDirection::EQ;
          break;
        case arith::CmpFPredicate::ONE:
        case arith::CmpFPredicate::UNE:
          direction = stablehlo::ComparisonDirection::NE;
          break;
        case arith::CmpFPredicate::OGT:
        case arith::CmpFPredicate::UGT:
          direction = stablehlo::ComparisonDirection::GT;
          break;
        case arith::CmpFPredicate::OGE:
        case arith::CmpFPredicate::UGE:
          direction = stablehlo::ComparisonDirection::GE;
          break;
        case arith::CmpFPredicate::OLT:
        case arith::CmpFPredicate::ULT:
          direction = stablehlo::ComparisonDirection::LT;
          break;
        case arith::CmpFPredicate::OLE:
        case arith::CmpFPredicate::ULE:
          direction = stablehlo::ComparisonDirection::LE;
          break;
        default:
          llvm_unreachable("Unsupported arith::CmpFPredicate for StableHLO conversion!");
      }
      return stablehlo::CompareOp::create(opBuilder, loc, hloO0, hloO1, direction, stablehlo::ComparisonType::FLOAT);
    })
    .Case<arith::CmpIOp>([&](arith::CmpIOp cmpOp){
      stablehlo::ComparisonDirection direction;
      bool isUnsigned = false;
      switch (cmpOp.getPredicate()) {
        case arith::CmpIPredicate::eq:
          direction = stablehlo::ComparisonDirection::EQ;
          break;
        case arith::CmpIPredicate::ne:
          direction = stablehlo::ComparisonDirection::NE;
          break;
        case arith::CmpIPredicate::sgt:
          direction = stablehlo::ComparisonDirection::GT;
          break;
        case arith::CmpIPredicate::ugt:
          direction = stablehlo::ComparisonDirection::GT;
          isUnsigned = true;
          break;
        case arith::CmpIPredicate::sge:
          direction = stablehlo::ComparisonDirection::GE;
          break;
        case arith::CmpIPredicate::uge:
          direction = stablehlo::ComparisonDirection::GE;
          isUnsigned = true;
          break;
        case arith::CmpIPredicate::slt:
          direction = stablehlo::ComparisonDirection::LT;
          break;
        case arith::CmpIPredicate::ult:
          direction = stablehlo::ComparisonDirection::LT;
          isUnsigned = true;
          break;
        case arith::CmpIPredicate::sle:
          direction = stablehlo::ComparisonDirection::LE;
          break;
        case arith::CmpIPredicate::ule:
          direction = stablehlo::ComparisonDirection::LE;
          isUnsigned = true;
          break;
        default:
          llvm_unreachable("Unsupported arith::CmpFPredicate for StableHLO conversion!");
      }
      if (isUnsigned) {
        return stablehlo::CompareOp::create(opBuilder, loc, hloO0, hloO1, direction, stablehlo::ComparisonType::UNSIGNED);
      }
      return stablehlo::CompareOp::create(opBuilder, loc, hloO0, hloO1, direction, stablehlo::ComparisonType::SIGNED);
    });
  assert(createdHLOOp->getNumResults() == 1);
  return createdHLOOp->getResult(0);
}

static Value handleUninaryArithOp(
  OpBuilder& opBuilder, 
  TranslationState& state, 
  Operation* op, 
  const llvm::SmallVector<Value>& args
) {
  assert(op->getNumResults() == 1 && op->getNumOperands() == 1);
  auto operand = args[0];
  auto loc = op->getLoc(); 
  auto createdHLOOp = llvm::TypeSwitch<Operation*, Operation*>(op)
    .Case([&](math::SinOp sop) {
      return stablehlo::SineOp::create(opBuilder, loc, operand, {});
    })
    .Case([&](math::ExpOp eop) {
      return stablehlo::ExpOp::create(opBuilder, loc, operand, {});
    })
    .Case([&](math::SqrtOp sop) {
      return stablehlo::SqrtOp::create(opBuilder, loc, operand, {});
    })
    .Case([&](arith::NegFOp nop) {
      return stablehlo::NegOp::create(opBuilder, loc, operand);
    })
    .Default([](auto) -> Operation* {
      llvm_unreachable("Unsupported");
    })
  ;
  assert(createdHLOOp->getNumResults() == 1);
  return createdHLOOp->getResult(0);
}

static Value handleArithOp(
  OpBuilder& opBuilder,
  TranslationState& state,
  Operation* arithOp,
  llvm::SmallVector<Value>& tensorArgs
) {
  return llvm::TypeSwitch<Operation*, Value>(arithOp)
    .Case<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, 
          arith::MulFOp, arith::MulIOp, arith::DivFOp, arith::DivSIOp,
          arith::CmpFOp, arith::CmpIOp
    >([&](Operation* binaryArithOp) {
      assert(tensorArgs.size() == 2);
      return handleBinaryArithOp(opBuilder, state, binaryArithOp, tensorArgs); 
    })
    .Case<math::SinOp, math::ExpOp, math::SqrtOp, arith::NegFOp>([&](Operation* unaryArithOp){
      assert(tensorArgs.size() == 1);
      return handleUninaryArithOp(opBuilder, state, unaryArithOp, tensorArgs);
    })
    .Case([&](arith::SelectOp sop){
      // %190 = "arith.select"(%189, %186, %187) : (i1, f64, f64) -> f64
      assert(sop.getNumOperands() == 3 && "Unexpected select oeprands size!");
      auto [trueTensor, falseTensor] = alignShapes(opBuilder, sop.getLoc(), tensorArgs[1], tensorArgs[2]);
      auto stableHLOSelectRes = stablehlo::SelectOp::create(opBuilder, sop.getLoc(), tensorArgs[0], trueTensor, falseTensor);
      return stableHLOSelectRes.getResult();
    })
  ;
}

struct ElementalState {
  TranslationState& globalState;
  llvm::DenseMap<Value, Value> localValueMap;

  ElementalState(TranslationState& state): globalState(state){};

  Value getTensorValue(OpBuilder &builder, Value value) {
    if (auto it = localValueMap.find(value);
        it != localValueMap.end()) {
      return it->second;
    }
    return globalState.getTensorValue(builder, value);
  }
};

static void translateElementalOp(
  OpBuilder& opBuilder,
  Operation* op,
  ElementalState& eleState
) {
  llvm::TypeSwitch<Operation*, void>(op)
    .Case([&](hlfir::DesignateOp designateOp){
      assert(llvm::all_of(designateOp.getIsTriplet(), [](bool isTriplet){return !isTriplet;}));
      auto memrefTensor = eleState.globalState.getTensorValue(opBuilder, designateOp.getMemref());
      eleState.localValueMap[designateOp.getResult()] = memrefTensor;
    })
    .Case([&](hlfir::ApplyOp applyOp){
      auto operandTensor = eleState.globalState.getTensorValue(opBuilder, applyOp.getOperand(0));
      eleState.localValueMap[applyOp.getResult()] = operandTensor;
    })
    .Case([&](fir::LoadOp loadOp){
      auto memrefTensor = eleState.getTensorValue(opBuilder, loadOp.getMemref());
      eleState.localValueMap[loadOp.getResult()] = memrefTensor;
    })
    .Case<
      arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, 
      arith::MulFOp, arith::MulIOp, arith::DivFOp, arith::DivSIOp,
      arith::CmpFOp, arith::CmpIOp,
      math::SinOp, math::ExpOp, math::SqrtOp, arith::NegFOp,
      arith::SelectOp
    >(
      [&](Operation* arithOp){
      // Reuse the handleArithOp
      assert(op->getNumResults() == 1);
      llvm::SmallVector<Value> tensorArgs;
      for (const auto& operand: op->getOperands()) {
        tensorArgs.push_back(eleState.getTensorValue(opBuilder, operand));
      }
      eleState.localValueMap[op->getResult(0)] = handleArithOp(opBuilder, eleState.globalState, arithOp, tensorArgs);
    })
    .Default([](Operation* unsupported){
      unsupported->dump();
      llvm_unreachable("unsupported");
    });
}

// INFO: this function only intends to cover elemental operation lowered from element-wise operations.
// For example: A = b * C, A = b + C, A = MAX(b, C)... where A, C are arrays, b is element.
// Example:
//    %13 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<1024x1024xf64> {
//    ^bb0(%arg17: index, %arg18: index):
//      %16 = hlfir.designate %7#0 (%arg17, %arg18)  : (!fir.ref<!fir.array<1024x1024xf64>>, index, index) -> !fir.ref<f64>
//      %17 = fir.load %16 : !fir.ref<f64>
//      %18 = arith.mulf %17, %12 fastmath<contract> : f64
//      hlfir.yield_element %18 : f64
//    }
static void translateElemental(
  OpBuilder& opBuilder, 
  TranslationState& state, 
  func::FuncOp newFunc, 
  hlfir::ElementalOp elementalOp,
  const llvm::DenseMap<Value, llvm::SmallVector<int64_t>>& sliceShiftMap
) {
  OpBuilder::InsertionGuard guard(opBuilder);
  opBuilder.setInsertionPointToEnd(&newFunc.front());

  assert(elementalOp.getRegion().hasOneBlock());
  auto& block = elementalOp.getRegion().getBlocks().front(); 
  ElementalState eleState(state);
  for (Operation& op: block.without_terminator()) {
    translateElementalOp(opBuilder, &op, eleState);
  }
  auto yieldOp = llvm::cast<hlfir::YieldElementOp>(block.getTerminator());
  assert(yieldOp);
  assert(eleState.localValueMap.contains(yieldOp.getElementValue()));
  Value yieldedTensor = eleState.localValueMap.at(yieldOp.getElementValue());
  state.valueMap[elementalOp.getResult()] = yieldedTensor;
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

      Value assignFromVal = state.getTensorValue(opBuilder, assignFrom);
      
      
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
    .Case<fir::AllocaOp, memref::AllocaOp>([&](auto allocaOp){
      state.referenceMap[allocaOp.getResult()] = {allocaOp.getResult(), {}};
    })
    .Case([&](hlfir::NoReassocOp nop){
      assert(state.valueMap.contains(nop.getOperand()));
      state.valueMap[nop.getResult()] = state.valueMap.at(nop.getOperand());
    })
    .Case<
      arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, 
      arith::MulFOp, arith::MulIOp, arith::DivFOp, arith::DivSIOp,
      arith::CmpFOp, arith::CmpIOp,
      math::SinOp, math::ExpOp, math::SqrtOp, arith::NegFOp,
      arith::SelectOp
    >(
      [&](Operation* arithOp){
      // Reuse the handleArithOp
      assert(op->getNumResults() == 1);
      llvm::SmallVector<Value> tensorArgs;
      for (const auto& operand: op->getOperands()) {
        tensorArgs.push_back(state.getTensorValue(opBuilder, operand));
      }
      state.valueMap[op->getResult(0)] = handleArithOp(opBuilder, state, arithOp, tensorArgs);
    })
    // INFO: built-in array operations
    .Case([&](hlfir::TransposeOp transposeOp){
      auto opTy = toCorrespondingTensorTy(transposeOp.getOperand().getType());
      auto resTy = RankedTensorType::get(
        llvm::SmallVector<int64_t>{opTy.getShape()[1], opTy.getShape()[0]},
        opTy.getElementType());
      auto stablehloTransposeOp = stablehlo::TransposeOp::create(
        opBuilder, op->getLoc(), resTy, state.getTensorValue(opBuilder, transposeOp.getOperand()),
        llvm::SmallVector<int64_t>{1, 0});
      state.valueMap[transposeOp.getResult()] = stablehloTransposeOp.getResult();
    })
    .Case([&](hlfir::MatmulOp mmOp){
      // %36 = hlfir.matmul %33#0 %35#0 {fastmath = #arith.fastmath<contract>}
      // : (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>) ->
      // !hlfir.expr<?x?xf64>
      auto op0 = state.getTensorValue(opBuilder, mmOp.getOperand(0));
      auto op1 = state.getTensorValue(opBuilder, mmOp.getOperand(1));
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
    .Case([&](hlfir::SumOp sumOp){
      assert(!sumOp.getMask() && "Masked SUM is not supported");
      
      // From:
      // %17 = "hlfir.sum"(%16, %2) : (!hlfir.expr<100x128xf64>, i32) -> !hlfir.expr<?xf64>
      // To: Reduce operation
      auto inputFir = sumOp.getArray();
      auto inputHlo = state.getTensorValue(opBuilder, inputFir);
      assert(inputHlo && "inputHlo not exist!");

      auto inputTy = llvm::dyn_cast<RankedTensorType>(inputHlo.getType());
      assert(inputTy && "hlfir.sum input must be a ranked tensor in StableHLO");
      auto eleTy = inputTy.getElementType();

      auto zeroAttr = opBuilder.getZeroAttr(eleTy);
      // accum of the sum result
      auto initValOp = stablehlo::ConstantOp::create(
          opBuilder, 
          op->getLoc(),
          DenseElementsAttr::get(RankedTensorType::get({}, eleTy), zeroAttr)
      );
      Value initVal = initValOp.getResult();

      llvm::SmallVector<int64_t> reduceDims;
      if (sumOp.getDim()) { // %2
        auto constOp = llvm::dyn_cast_or_null<arith::ConstantOp>(sumOp.getDim().getDefiningOp());
        assert(constOp && "the dim must be known!");
        int64_t fortranDimVal = llvm::cast<IntegerAttr>(constOp.getValue()).getInt();
        int64_t stableHloDim = inputTy.getRank() - fortranDimVal;
        assert(fortranDimVal >= 1);
        assert(fortranDimVal <= inputTy.getRank());
        reduceDims.push_back(stableHloDim);
      } else {
        for (int64_t i = 0; i < inputTy.getRank(); ++i) {
          reduceDims.push_back(i);
        }
      }
      auto resultFirTy = sumOp.getResult().getType();
      auto resultHloTy = toCorrespondingTensorTy(resultFirTy);
      auto reduceOp = stablehlo::ReduceOp::create(
          opBuilder, 
          op->getLoc(), 
          TypeRange{resultHloTy},
          ValueRange{inputHlo}, 
          ValueRange{initVal},
          opBuilder.getDenseI64ArrayAttr(reduceDims)
      );
      Region &region = reduceOp.getBody();
      Block *block = opBuilder.createBlock(&region);
      auto scalarTy = RankedTensorType::get({}, eleTy);
      block->addArguments({scalarTy, scalarTy}, {op->getLoc(), op->getLoc()});

      OpBuilder::InsertionGuard reduceBlockGuard(opBuilder);
      opBuilder.setInsertionPointToStart(block);
      auto addOp = stablehlo::AddOp::create(
        opBuilder, 
        op->getLoc(),
        scalarTy, 
        block->getArgument(0),
        block->getArgument(1)
      );
      stablehlo::ReturnOp::create(
        opBuilder, 
        op->getLoc(),
        ValueRange{addOp.getResult()}
      );
      state.valueMap[sumOp.getResult()] = reduceOp.getResult(0);
    })
    .Case([&](hlfir::DotProductOp dpOp){
      // example: %151 = hlfir.dot_product %148#0 %150#0
      // TODO: only support 1d arrays dot product
      assert(dpOp.getNumOperands() == 2 && "Failure of expecting dot product op has 2 operands!\n");
      auto lhs = state.getTensorValue(opBuilder, dpOp.getOperand(0));
      auto rhs = state.getTensorValue(opBuilder, dpOp.getOperand(1));
     
      auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
      auto scalarType = RankedTensorType::get({}, lhsTy.getElementType());
      stablehlo::DotDimensionNumbersAttr attr =
          stablehlo::DotDimensionNumbersAttr::get(opBuilder.getContext(), {}, {}, {0}, {0});
      ArrayAttr precisionConfig = {};
      stablehlo::DotAlgorithmAttr algoAttr = {};
      auto stablehloDotProductOp = stablehlo::DotGeneralOp::create(
        opBuilder, 
        op->getLoc(), 
        scalarType, 
        lhs, 
        rhs, 
        attr,
        precisionConfig, 
        algoAttr
      );
      state.valueMap[dpOp.getResult()] = stablehloDotProductOp.getResult();
    })
    // INFO: support especially for scalar operation
    .Case([&](fir::ZeroOp zeroOp){
      mlir::Type resType = zeroOp.getType();
      mlir::Attribute zeroAttr;
      if (resType.isIndex()) {
        zeroAttr = IntegerAttr::get(IntegerType::get(opBuilder.getContext(), 64), 0);
      } else if (llvm::isa<mlir::IntegerType>(resType)) {
        zeroAttr = IntegerAttr::get(resType, 0);
      } else if (llvm::isa<mlir::FloatType>(resType)) {
        zeroAttr = FloatAttr::get(resType, 0.0);
      } else {
        zeroAttr = opBuilder.getZeroAttr(resType);
      }
      auto stablehloZeroOp = stablehlo::ConstantOp::create(opBuilder, zeroOp->getLoc(), zeroAttr);
      state.valueMap[zeroOp.getResult()] = stablehloZeroOp.getResult();
    })
    .Case([&](fir::DeclareOp declareOp){
      assert(state.referenceMap.contains(declareOp.getMemref()));
      state.referenceMap[declareOp.getResult()] = state.referenceMap.at(declareOp.getMemref()); 
    })
    .Case([&](fir::ConvertOp convertOp){
      auto isReferenceLike = [](Type type) -> bool {
        return fir::isa_ref_type(type) 
          || fir::isa_box_type(type)
          || llvm::isa<MemRefType>(type);
      };
      // If input element type and output element type is the same, should not generate any stablehlo convert.
      // For example:
      // - %10 = fir.convert %4 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
      auto convertFrom = convertOp.getOperand();
      auto convertTo = convertOp.getResult();
      // both from and to are mems
      if (isReferenceLike(convertFrom.getType()) && isReferenceLike(convertTo.getType())) {
        assert(state.referenceMap.contains(convertFrom));
        state.referenceMap[convertTo] = state.referenceMap.at(convertFrom);
      } else {
        assert(!isReferenceLike(convertFrom.getType()));
        assert(!isReferenceLike(convertTo.getType()));
        auto input = state.getTensorValue(opBuilder, convertFrom);
        assert(llvm::isa<RankedTensorType>(input.getType()));
        Type outputType = convertTo.getType();
        auto stableHLOConvertOp = stablehlo::ConvertOp::create(opBuilder, op->getLoc(), input, outputType);
        state.valueMap[convertTo] = stableHLOConvertOp.getResult();
      }
    })
    .Case([&](::mlir::affine::AffineLoadOp loadOp){
      // TODO: loadOp's mem can have indices? should create a slice?
      assert(loadOp.getNumOperands() == 1);
      assert(loadOp.getIndices().empty());
      state.valueMap[loadOp.getResult()] = state.getTensorValue(opBuilder, loadOp.getMemref());
    })
    .Case([&](::mlir::bufferization::ToTensorOp toTensorOp){
      state.valueMap[toTensorOp.getResult()] = state.getTensorValue(opBuilder, toTensorOp.getOperand());
    })
    .Case<affine::AffineStoreOp>([&](affine::AffineStoreOp storeOp){
      // TODO: can have indices      
      assert(storeOp.getIndices().empty());
      auto storeFrom = storeOp.getValueToStore(); 
      auto storeTo = storeOp.getMemRef();
      assert(state.valueMap.contains(storeFrom));
      auto storeToRef = state.referenceMap.at(storeTo);
      assert(storeToRef.triplets.empty());
      state.memoryMap[storeToRef.root] = state.getTensorValue(opBuilder, storeFrom);
    })
    .Case<memref::StoreOp>([&](memref::StoreOp storeOp){
      // TODO: can have indices      
      assert(storeOp.getIndices().empty());
      auto storeFrom = storeOp.getValueToStore(); 
      auto storeTo = storeOp.getMemRef();
      assert(state.valueMap.contains(storeFrom));
      auto storeToRef = state.referenceMap.at(storeTo);
      assert(storeToRef.triplets.empty());
      state.memoryMap[storeToRef.root] = state.getTensorValue(opBuilder, storeFrom);
    })
    .Case([&](bufferization::MaterializeInDestinationOp mop){
      assert(state.referenceMap.contains(mop.getDest()));
      auto destRefVal = state.referenceMap.at(mop.getDest());
      assert(destRefVal.triplets.empty());
      state.memoryMap[destRefVal.root] = state.getTensorValue(opBuilder, mop.getSource());
    })
    .Case([&](bufferization::ToBufferOp toBufferOp){
      state.referenceMap[toBufferOp.getResult()] = {toBufferOp.getResult(), {}};
      state.memoryMap[toBufferOp.getResult()] = state.getTensorValue(opBuilder, toBufferOp.getOperand());
    })
    .Case([&](func::CallOp callOp){
      // NOTE: this operation is generated when outline the scalar loop, it will be replaced with a function call in the main function.
      // In our outline pass design, this function's arguments count and returned values count will be strictly equal.
      auto paramsCount = callOp.getNumOperands();
      auto originalArgs = callOp.getArgOperands();
      llvm::SmallVector<Value> translatedArgs(paramsCount);
      for (int i = 0; i < paramsCount; i++) {
        Value originalArg = originalArgs[i];
        translatedArgs[i] = state.getTensorValue(opBuilder, originalArg);
      }
      auto moduleOp = callOp->getParentOfType<ModuleOp>();
      assert(moduleOp);
      func::FuncOp calleeFunc = moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
      assert(calleeFunc);
      auto translatedCallOp = func::CallOp::create(opBuilder, callOp.getLoc(), calleeFunc, translatedArgs);
      assert(callOp.getNumOperands() == callOp.getNumResults());
      assert(callOp.getNumResults() == translatedCallOp.getNumResults());
      for (int i = 0; i < callOp.getNumOperands(); i++) {
        state.valueMap[callOp.getResult(i)] = translatedCallOp.getResult(i);
      }
    })
    // INFO: ignored results, instead of print out
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


  bool isStableHLOFunction(func::FuncOp funcOp) {
    return llvm::any_of(funcOp.front().getOperations(), [](Operation& op){
      return op.getName().getDialectNamespace().contains_insensitive("stablehlo");
    });
  }

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
      if (!isStableHLOFunction(fOp)) {
        funcsToReplace.push_back(fOp);
      }
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
