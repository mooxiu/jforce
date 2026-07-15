/// TODO: If I have time, I want to rewrite this.


#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <alloca.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <utility>
#include "../support/profiler.h"
#include "../support/utilities.h"
#include "Utils.h"

using namespace mlir;

namespace {
/// valueMap, argsTrackingMaps are 2 maps we'll keep updating when scanning
/// - valueMap: tracking the each operand of FIR pointing to the value of each
/// operand in Stablehlo function.
/// - argsTrackingMap: tracking the current value of arguments of stablehlo
/// pointing to, practically a reverse map of `valueMap`.
struct TrackingInfo {
private:
  void printMap(IRMapping map, ::mlir::StringRef str) {
    llvm::dbgs() << "[DEBUG] " << str << ":\n";
    for (const auto& entry: valueMap.getValueMap()) {
      llvm::dbgs() << "    > ";
      entry.getFirst().printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << ": ";
      entry.getSecond().printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n";
  }
public:
  // Key: value in FIR function
  // Value: value in StableHLO function
  // FIXME: this map should only track from "original MLIR canonical mem" to "StableHLO MLIR tensor"
  IRMapping valueMap;

  // FIXME: this map should only tracking from "original MLIR mem" to "original MLIR canonical mem"
  IRMapping aliasMap;

  // Key: value of one of StableHLO function's arguments
  // Value: value in FIR function
  IRMapping argsTrackingMap;

  void debugPrintValueMap() {
    printMap(valueMap, "Value Map");
  }

  void debugPrintArgsTrackingMap() {
    printMap(argsTrackingMap, "Arg Tracking Map");
  }
};


enum OperationType {
  CREATE_VAL,
  CREATE_MEM,
  READ_VAL_FROM_MEM,
  WRITE_VAL_TO_MEM,
  VAL_TO_VAL,
  MEM_TO_MEM,
};

// FIXME: this is a ad-hoc fix, need to add alias map to tracking
static void mapMemAndAliasChain(TrackingInfo &tracking, Value mem, Value hloVal) {
  if (!mem) return;

  tracking.valueMap.map(mem, hloVal);
  Operation *defOp = mem.getDefiningOp();

  if (!defOp) return;
  if (auto convOp = llvm::dyn_cast<fir::ConvertOp>(defOp)) {
    mapMemAndAliasChain(tracking, convOp.getOperand(), hloVal);
    return;
  }
  if (auto declOp = llvm::dyn_cast<fir::DeclareOp>(defOp)) {
    mapMemAndAliasChain(tracking, declOp.getOperand(0), hloVal);
    return;
  }
  if (auto declOp = llvm::dyn_cast<hlfir::DeclareOp>(defOp)) {
    mapMemAndAliasChain(tracking, declOp.getOperand(0), hloVal);
    return;
  }
  if (auto castOp = llvm::dyn_cast<memref::CastOp>(defOp)) {
    mapMemAndAliasChain(tracking, castOp.getSource(), hloVal);
    return;
  }
}

template<OperationType Ty>
static void updateTracking(
  TrackingInfo& tracking, 
   mlir::ValueRange oldOpFromVars,
   mlir::ValueRange oldOpToVars,
   mlir::ValueRange newOpToVars
) {
  switch (Ty) {
    case CREATE_VAL:
      // example: constantOp
      // should creating corresponding values on stablehlo function
      assert(oldOpFromVars.empty());
      assert(oldOpToVars.size() > 0);
      assert(oldOpToVars.size() == newOpToVars.size());
      for (int i = 0 ; i < oldOpToVars.size(); i++) {
        tracking.valueMap.map(
            oldOpToVars[i], newOpToVars[i]
          );
      }
      break;
    case CREATE_MEM:
      // example: allocaOp
      // usually do not need to do anything, just ignore.
      break;
    case READ_VAL_FROM_MEM:
      // example: loadOp
      // pointing current value to the value that this memory is pointed to, no need to create corresponding statements in stableHLO.
      assert(oldOpFromVars.size() == oldOpToVars.size());
      for (int i = 0; i < oldOpToVars.size(); i++) {
        auto mem = oldOpFromVars[i];
        auto val = oldOpToVars[i];
        if (tracking.valueMap.contains(mem)) {
          tracking.valueMap.map(
              val,
              tracking.valueMap.lookup(mem)
            );
        } else {
          // Possibly reading from a dummy
          // DO NOTHING
        } 
      }
      break;
    case WRITE_VAL_TO_MEM:
      // example: storeOp
      //
      // example: bufferization.materialize_in_destination %9#1 in writable %alloca : (tensor<f64>, memref<f64>) -> ()
      // - val: %9#1, should have corresponding stablehlo value: valHLO
      // - mem: %alloca
      //
      assert(oldOpFromVars.size() == oldOpToVars.size());
      for (int i = 0; i < oldOpFromVars.size(); i++) {
        auto val = oldOpFromVars[i];
        auto mem = oldOpToVars[i];
        assert(tracking.valueMap.contains(val));
        auto valHLO = tracking.valueMap.lookup(val);

        if (tracking.valueMap.contains(mem)) {
          auto memHLO = tracking.valueMap.lookup(mem); 
          tracking.argsTrackingMap.map(memHLO, valHLO);
        }
        // this is a mem has not written to anything
        mapMemAndAliasChain(tracking, mem, valHLO);
      }
      break;
    case VAL_TO_VAL:
      // example: some fir::convert, stablehlo function's inputs and outputs
      assert(oldOpToVars.size() == newOpToVars.size());
      for (int i = 0; i < oldOpToVars.size(); i++) {
        tracking.valueMap.map(oldOpToVars[i], newOpToVars[i]);
      }
      break;
    case MEM_TO_MEM:
      // exmaple: some fir::convert, hlfir.declare, fir.declare
      assert(oldOpFromVars.size() == oldOpToVars.size());
      for (int i = 0; i < oldOpToVars.size(); i++) {
        auto memFrom = oldOpFromVars[i];
        auto memTo = oldOpToVars[i];
        if (tracking.valueMap.contains(memFrom)) {
          tracking.valueMap.map(
            memTo,
            tracking.valueMap.lookup(memFrom)
          );
        }
      }
      break;
    default:
      llvm::errs() << "Unexepected Enum Value!\n";
      std::exit(EXIT_FAILURE);
  }
}

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

/// We're dealing with TargetOp like following:
/// > omp.target map_entries(%141 -> %arg0, %142 -> %arg1, %145 -> %arg2 :
/// !fir.ref<!fir.array<10xf32>>, !fir.ref<f32>, !fir.ref<!fir.array<10xf32>>) {
///
/// In which, "%141, %142, %145" is out values, they will be used when get the
/// value and call stablehlo function;
/// "%arg0, %arg1, %arg2" are the values we need to track.
static func::FuncOp createFunction(mlir::MLIRContext *context,
                                   TrackingInfo &tracking,
                                   func::FuncOp inputOp) {
  auto &firstRegion = inputOp->getRegion(0);
  auto &block = firstRegion.getBlocks().front();

  mlir::SmallVector<Type> argsTypes;
  for (const auto &arg : block.getArguments()) {
    argsTypes.push_back(toCorrespondingTensorTy(arg.getType()));
  }

  auto funcType = mlir::FunctionType::get(context, argsTypes, argsTypes);
  // Has to be named as `main` to be compiled by XLA.
  auto funcOp = func::FuncOp::create(inputOp->getLoc(), "main", funcType, {});
  // we need to update the valueMap!
  funcOp.addEntryBlock();

  for (int i = 0; i < block.getNumArguments(); i++) {
    Value firFuncArg = block.getArgument(i);
    Value stablehloFuncArg = funcOp.getArgument(i);
    auto argAttr = inputOp.getArgAttrOfType<IntegerAttr>(i, JIT_LITERAL_VAL_ATTR_NAME);
    if (argAttr) {
      funcOp.setArgAttr(i, JIT_LITERAL_VAL_ATTR_NAME, argAttr);
    }
    tracking.valueMap.map(firFuncArg, stablehloFuncArg);
    tracking.argsTrackingMap.map(stablehloFuncArg, stablehloFuncArg);
  }
  return funcOp;
}

static void handleArithUnaryOp(TrackingInfo &tracking, OpBuilder &opBuilder,
                               func::FuncOp &funcOp, Operation *unaryOp) {
  assert(unaryOp->getNumResults() == 1 && unaryOp->getNumOperands() == 1);
  auto operand = unaryOp->getOperand(0);
  auto result = unaryOp->getResult(0);
  auto operandSrc = tracking.valueMap.lookup(operand);
  assert(operandSrc && "OperandSource should exist!");
  llvm::TypeSwitch<Operation *>(unaryOp)
      .Case([&](math::SinOp sop) {
        // should have same type
        auto resTy = toCorrespondingTensorTy(operandSrc.getType());
        auto sinOp = stablehlo::SineOp::create(opBuilder, funcOp.getLoc(),
                                               resTy, operandSrc, {});
        tracking.valueMap.map(result, sinOp.getResult());
      })
      .Case([&](math::ExpOp eop) {
        auto resTy = toCorrespondingTensorTy(operandSrc.getType());
        auto stablehloExOp = stablehlo::ExpOp::create(
            opBuilder, funcOp.getLoc(), resTy, operandSrc, {});
        tracking.valueMap.map(result, stablehloExOp.getResult());
      })
      .Case([&](math::SqrtOp sop) {
        auto resTy = toCorrespondingTensorTy(operandSrc.getType());
        auto stablehloSqrtOp = stablehlo::SqrtOp::create(
            opBuilder, funcOp.getLoc(), resTy, operandSrc, {});
        tracking.valueMap.map(result, stablehloSqrtOp.getResult());
      })
      .Case([&](arith::NegFOp nop) {
        auto resTy = toCorrespondingTensorTy(operandSrc.getType());
        auto stablehloNegOp = stablehlo::NegOp::create(
            opBuilder, funcOp.getLoc(), resTy, operandSrc);
        tracking.valueMap.map(result, stablehloNegOp.getResult());
      });
}

/// Only support automatic dimension broadcasting right now, for example:
/// - tensor<f32> -> tensor<10xf32>
/// - tensor<10xf32> -> tensor<10x10xf32>
///
/// Ref: https://openxla.org/stablehlo/spec#broadcast_in_dim
static void handleArithBinaryOp(
  TrackingInfo &tracking,
  OpBuilder& opBuilder,
  func::FuncOp funcOp,
  Operation* op
) {
  assert(op->getNumOperands() == 2 && op->getNumResults() == 1);
  auto srcOperand0 = op->getOperand(0);
  auto srcOperand1 = op->getOperand(1);
  assert(tracking.valueMap.contains(srcOperand0));
  assert(tracking.valueMap.contains(srcOperand1));
  auto alignShape = [&](Value hloO0, Value hloO1) -> std::pair<Value, Value> {
    assert(llvm::isa<RankedTensorType>(hloO1.getType()));
    assert(llvm::isa<RankedTensorType>(hloO1.getType()));
    auto typeInfo0 = inspectTypeInfo(hloO0.getType());
    auto typeInfo1 = inspectTypeInfo(hloO1.getType());
    if (typeInfo0.rank == typeInfo1.rank) {
      assert(typeInfo0.shape == typeInfo1.shape);
      return std::pair<Value, Value>(hloO0, hloO1);
    }
    if (typeInfo0.rank < typeInfo1.rank) {
      auto broadcastOp = stablehlo::BroadcastInDimOp::create(
          opBuilder, 
          funcOp.getLoc(),
          hloO1.getType(),
          hloO0,
          opBuilder.getDenseI64ArrayAttr({})
        );
      return std::make_pair(broadcastOp.getResult(), hloO1);
    } else {
      // hlo0 has larger rank
      auto broadcastOp = stablehlo::BroadcastInDimOp::create(
          opBuilder, 
          funcOp.getLoc(),
          hloO0.getType(),
          hloO1,
          opBuilder.getDenseI64ArrayAttr({})
        );
      return std::make_pair(hloO0, broadcastOp.getResult());
    }
  };
  auto opPair = alignShape(tracking.valueMap.lookup(srcOperand0), tracking.valueMap.lookup(srcOperand1));
  auto hloO0 = opPair.first;
  auto hloO1 = opPair.second;
       // .Case<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, 
       //      arith::MulFOp, arith::AddIOp, arith::DivFOp, arith::DivSIOp,
       //      arith::CmpFOp, arith::CmpIOp>([&](auto arithBinaryOp) {
  Operation* createdHLOOp = llvm::TypeSwitch<Operation*, Operation*>(op)
    .Case<arith::AddFOp, arith::AddIOp>([&](auto){
      auto resTy = hloO0.getType();
      return stablehlo::AddOp::create(opBuilder, funcOp.getLoc(), resTy, hloO0, hloO1);
    })
    .Case<arith::SubFOp, arith::SubIOp>([&](auto){
      auto resTy = hloO0.getType();
      return stablehlo::SubtractOp::create(opBuilder, funcOp.getLoc(), resTy, hloO0, hloO1);
    })
    .Case<arith::MulFOp, arith::MulIOp>([&](auto){
      auto resTy = hloO0.getType();
      return stablehlo::MulOp::create(opBuilder, funcOp.getLoc(), resTy, hloO0, hloO1);
    })
    .Case<arith::DivFOp, arith::DivSIOp>([&](auto){
      auto resTy = hloO0.getType();
      return stablehlo::DivOp::create(opBuilder, funcOp.getLoc(), resTy, hloO0, hloO1);
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
      return stablehlo::CompareOp::create(opBuilder, funcOp.getLoc(), hloO0, hloO1, direction, stablehlo::ComparisonType::FLOAT);
    })
    .Case<arith::CmpIOp>([&](arith::CmpIOp cmpOp){
      stablehlo::ComparisonDirection direction;
      switch (cmpOp.getPredicate()) {
        case arith::CmpIPredicate::eq:
          direction = stablehlo::ComparisonDirection::EQ;
          break;
        case arith::CmpIPredicate::ne:
          direction = stablehlo::ComparisonDirection::NE;
          break;
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::ugt:
          direction = stablehlo::ComparisonDirection::GT;
          break;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::uge:
          direction = stablehlo::ComparisonDirection::GE;
          break;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ult:
          direction = stablehlo::ComparisonDirection::LT;
          break;
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::ule:
          direction = stablehlo::ComparisonDirection::LE;
          break;
        default:
          llvm_unreachable("Unsupported arith::CmpFPredicate for StableHLO conversion!");
      }
      return stablehlo::CompareOp::create(opBuilder, funcOp.getLoc(), hloO0, hloO1, direction, stablehlo::ComparisonType::SIGNED);
    })
  ;
  // updateTracking<OperationType Ty>(TrackingInfo &tracking, mlir::ValueRange oldOpFromVars, mlir::ValueRange oldOpToVars, mlir::ValueRange newOpToVars)
  updateTracking<VAL_TO_VAL>(tracking, op->getOperands(), op->getResults(), createdHLOOp->getResults());                        
  return;
}

static void handleBuiltinOperators(TrackingInfo &tracking, OpBuilder &opBuilder, func::FuncOp &funcOp, Operation *op) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<hlfir::DotProductOp>([&](hlfir::DotProductOp dpOp) {
        // example: %151 = hlfir.dot_product %148#0 %150#0
        // TODO: only support 1d arrays dot product
        assert(dpOp.getNumOperands() == 2 &&
               "Failure of expecting dot product op has 2 operands!\n");
        auto lhs = tracking.valueMap.lookup(dpOp.getOperand(0));
        assert(lhs && "lhs not exist!");
        auto rhs = tracking.valueMap.lookup(dpOp.getOperand(1));
        assert(rhs && "rhs not exist!");

        auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
        if (!lhsTy) {
          llvm::errs() << "Unexpected lhs type!\n";
          exit(1);
        }
        auto scalarType = RankedTensorType::get({}, lhsTy.getElementType());
        stablehlo::DotDimensionNumbersAttr attr =
            stablehlo::DotDimensionNumbersAttr::get(funcOp.getContext(), {}, {},
                                                    {0}, {0});
        ArrayAttr precisionConfig = {};
        stablehlo::DotAlgorithmAttr algoAttr = {};
        auto stablehloDotProductOp = stablehlo::DotGeneralOp::create(
            opBuilder, funcOp->getLoc(), scalarType, lhs, rhs, attr,
            precisionConfig, algoAttr);
        tracking.valueMap.map(dpOp.getResult(),
                              stablehloDotProductOp.getResult());
      })
      .Case<hlfir::MatmulOp>([&](hlfir::MatmulOp mmOp) {
        // %36 = hlfir.matmul %33#0 %35#0 {fastmath = #arith.fastmath<contract>}
        // : (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>) ->
        // !hlfir.expr<?x?xf64>
        auto op0 = tracking.valueMap.lookup(mmOp.getOperand(0));
        assert(op0 && "op0 not exist!");
        auto op1 = tracking.valueMap.lookup(mmOp.getOperand(1));
        assert(op1 && "op1 not exist!");
        auto op0Ty = toCorrespondingTensorTy(op0.getType());
        auto op1Ty = toCorrespondingTensorTy(op1.getType());
        auto resTy = RankedTensorType::get(
            llvm::SmallVector<int64_t>{op1Ty.getShape()[0],
                                       op0Ty.getShape()[1]},
            op0Ty.getElementType());

        mlir::ArrayAttr config = {};
        mlir::stablehlo::DotAlgorithmAttr algo = {};
        auto dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
            funcOp.getContext(), SmallVector<int64_t>{}, SmallVector<int64_t>{},
            SmallVector<int64_t>{1}, SmallVector<int64_t>{0});

        auto stablehloDotGeneralOp = stablehlo::DotGeneralOp::create(
            opBuilder, funcOp.getLoc(), resTy, op1, op0, dims, config, algo);
        tracking.valueMap.map(mmOp.getResult(),
                              stablehloDotGeneralOp.getResult());
      })
      .Case<hlfir::TransposeOp>([&](hlfir::TransposeOp tOp) {
        // %24 = hlfir.transpose %23#0 : (!fir.box<!fir.array<?x?xf64>>) ->
        // !hlfir.expr<?x?xf64>
        auto op = tracking.valueMap.lookup(tOp.getOperand());
        assert(op && "op not exist!");
        auto opTy = toCorrespondingTensorTy(op.getType());
        auto resTy = RankedTensorType::get(
            llvm::SmallVector<int64_t>{opTy.getShape()[1], opTy.getShape()[0]},
            opTy.getElementType());
        auto stablehloTransposeOp = stablehlo::TransposeOp::create(
            opBuilder, funcOp.getLoc(), resTy, op,
            llvm::SmallVector<int64_t>{1, 0});
        tracking.valueMap.map(tOp.getResult(),
                              stablehloTransposeOp.getResult());
      })
      .Case<hlfir::SumOp>([&](hlfir::SumOp sumOp) {
        // %17 = "hlfir.sum"(%16, %2) <{fastmath = #arith.fastmath<contract>,
        // operandSegmentSizes = array<i32: 1, 1, 0>}> :
        // (!hlfir.expr<100x128xf64>, i32) -> !hlfir.expr<?xf64>
        auto inputFir = sumOp.getArray();
        auto inputHlo = tracking.valueMap.lookup(inputFir);
        assert(inputHlo && "inputHlo not exist!");

        auto inputTy = llvm::dyn_cast<RankedTensorType>(inputHlo.getType());
        assert(inputTy &&
               "hlfir.sum input must be a ranked tensor in StableHLO");
        auto eleTy = inputTy.getElementType();

        auto zeroAttr = opBuilder.getZeroAttr(eleTy);
        // accum of the sum result
        auto initValOp = stablehlo::ConstantOp::create(
            opBuilder, funcOp.getLoc(),
            DenseElementsAttr::get(RankedTensorType::get({}, eleTy), zeroAttr));
        Value initVal = initValOp.getResult();

        llvm::SmallVector<int64_t> reduceDims;
        if (sumOp.getDim()) { // %2
          auto constOp = llvm::dyn_cast_or_null<arith::ConstantOp>(
              sumOp.getDim().getDefiningOp());
          assert(constOp && "the dim must be known!");
          int64_t fortranDimVal =
              llvm::cast<IntegerAttr>(constOp.getValue()).getInt();
          int64_t stableHloDim = inputTy.getRank() - fortranDimVal;
          reduceDims.push_back(stableHloDim);
        } else {
          for (int64_t i = 0; i < inputTy.getRank(); ++i) {
            reduceDims.push_back(i);
          }
        }

        auto resultFirTy = sumOp.getResult().getType();
        auto resultHloTy = toCorrespondingTensorTy(resultFirTy);

        auto reduceOp = stablehlo::ReduceOp::create(
            opBuilder, funcOp.getLoc(), TypeRange{resultHloTy},
            ValueRange{inputHlo}, ValueRange{initVal},
            opBuilder.getDenseI64ArrayAttr(reduceDims));

        auto storedInsertPoint = opBuilder.saveInsertionPoint();
        Region &region = reduceOp.getBody();
        Block *block = opBuilder.createBlock(&region);
        auto scalarTy = RankedTensorType::get({}, eleTy);

        block->addArguments({scalarTy, scalarTy},
                            {funcOp.getLoc(), funcOp.getLoc()});

        opBuilder.setInsertionPointToStart(block);

        auto addOp = stablehlo::AddOp::create(opBuilder, funcOp.getLoc(),
                                              scalarTy, block->getArgument(0),
                                              block->getArgument(1));
        stablehlo::ReturnOp::create(opBuilder, funcOp.getLoc(),
                                    ValueRange{addOp.getResult()});
        opBuilder.restoreInsertionPoint(storedInsertPoint);
        tracking.valueMap.map(sumOp.getResult(), reduceOp.getResult(0));
      })
      .Default([](auto) {
        llvm::errs() << "Not Supported Builtin Operators!\n";
        std::exit(EXIT_FAILURE);
      });
}

// If input element type and output element type is the same, should not generate any stablehlo convert.
// For example:
// - %10 = fir.convert %4 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
static void handleConvertOp(TrackingInfo &tracking, OpBuilder &opBuilder, func::FuncOp &funcOp, fir::ConvertOp convertOp) {
  
  auto convertFrom = convertOp.getOperand();
  auto convertTo = convertOp.getResult();
  auto convertFromTypeInfo = inspectTypeInfo(convertFrom.getType());
  auto convertToTypeInfo = inspectTypeInfo(convertTo.getType());

  if (convertFromTypeInfo.elementTy == convertToTypeInfo.elementTy) {
    // INFO: mem conversion
    updateTracking<OperationType::MEM_TO_MEM>(tracking, {convertFrom}, {convertTo}, {});
  } else {
    auto stableHLOConvertOp = stablehlo::ConvertOp::create(opBuilder, funcOp.getLoc(), convertFrom, convertTo.getType());
    updateTracking<OperationType::VAL_TO_VAL>(tracking, {convertFrom}, {convertTo}, {stableHLOConvertOp.getResult()});
  }
}

/// DesignateOp can generate slice
static void handleDesignateOp(
    TrackingInfo &tracking, OpBuilder &opBuilder, func::FuncOp funcOp,
    hlfir::DesignateOp designateOp,
    const llvm::DenseMap<Value, llvm::SmallVector<int64_t>> &sliceShiftMap) {
  auto isTriplet = designateOp.getIsTriplet();
  auto resultOperand = designateOp.getResult();
  auto memRef = designateOp.getMemref();

  bool isSlicing = [&]() {
    auto containsSlicing = false;
    for (bool tri : isTriplet) {
      containsSlicing = tri || containsSlicing;
      if (containsSlicing) {
        return true;
      }
    }
    return containsSlicing;
  }();

  if (!isSlicing) {
    // example: %451 = "hlfir.designate"(%447#0, %arg9)
    // Often inside of an elemental operation
    auto memRefV = tracking.valueMap.lookup(memRef);
    assert(memRefV && "memRefV not exist!");
    tracking.valueMap.map(resultOperand, memRefV);
  } else {
    // example 1: 1D slicing %1#0 [%c1: %c5: %c1]
    // - %7 = hlfir.designate %1#0 (%c1:%c5:%c1)  shape %4 :
    // (!fir.box<!fir.array<?xf64>>, index, index, index, !fir.shape<1>) ->
    // !fir.box<!fir.array<?xf64>>
    //
    // example 2: 2D slicing %6#0 []
    //- %8 = "hlfir.designate"(%6#0, %1, %2, %1, %1, %0, %1, %7) <{is_triplet =
    //array<i1: true, true>, operandSegmentSizes = array<i32: 1, 0, 6, 0, 1,
    //0>}> : (!fir.box<!fir.array<5x5xf64>>, index, index, index, index, index,
    //index, !fir.shape<2>) -> !fir.box<!fir.array<2x5xf64>>
    auto indices = designateOp.getIndices();
    assert(indices.size() >= 3 && indices.size() % 3 == 0 &&
           "Unexpected Indices!");
    auto rank = indices.size() / 3;

    // index shifts
    auto defOp = llvm::dyn_cast<hlfir::DeclareOp>(
        designateOp.getMemref().getDefiningOp());
    assert(defOp && "Defining Op of designateOp memref should be a declareOp!");

    llvm::SmallVector<int64_t> defaultShifts(rank, 1);
    const llvm::SmallVector<int64_t> *shifts;
    auto ssOp =
        llvm::dyn_cast<fir::ShapeShiftOp>(defOp.getShape().getDefiningOp());
    if (ssOp) {
      auto it = sliceShiftMap.find(ssOp.getResult());
      assert(it != sliceShiftMap.end() &&
             "All shapeshifts should already have been recorded!");
      shifts = &(it->getSecond());
    } else {
      shifts = &defaultShifts;
    }
    assert((shifts->size() == rank) ||
           (llvm::errs() << "Shifts size should be the same with dimension "
                            "size! While shifts size: "
                         << shifts->size() << ", while rank =" << rank,
            false));

    llvm::SmallVector<int64_t> sliceStartIdxVals;
    llvm::SmallVector<int64_t> sliceLimitIdxVals;
    llvm::SmallVector<int64_t> sliceStrideIdxVals;

    auto getI64Val = [](const mlir::Value &idxVal) -> int64_t {
      arith::ConstantIndexOp constOp =
          llvm::dyn_cast<arith::ConstantIndexOp>(idxVal.getDefiningOp());
      if (!constOp) {
        llvm::errs() << "This is not constantOp, you stupid!\n";
        std::exit(EXIT_FAILURE);
      }
      return constOp.value();
    };

    // example, FortranSlice(-1, 2, 1) with shift -1, should be FortranSlice(1,
    // 4, 1), should be stablehlo.slice(0, 4, 1) example, FortranSlice(1, 2, 1)
    // with shift 0, should be FortranSlice(1, 2, 1), should be
    // stablehlo.slice(0, 3, 1)
    for (int i = 0; i < rank; i++) {
      auto offSet = i * 3;

      auto startIdxVal = getI64Val(indices[offSet]) - (*shifts)[i];
      auto limitIdxVal = getI64Val(indices[offSet + 1]) - (*shifts)[i] +
                         1; // In stablehlo, the limit idx is not included
      assert(getI64Val(indices[offSet + 2]) == 1 &&
             "Only dealing with stride = 1 for now!!!");
      auto strideIdxVal = getI64Val(indices[offSet + 2]);

      sliceStartIdxVals.push_back(startIdxVal);
      sliceLimitIdxVals.push_back(limitIdxVal);
      sliceStrideIdxVals.push_back(strideIdxVal);
    }

    // create slice operation
    // we're not inserting in place, so donot set the insertion point of
    // opBuilder

    // Debugging...
    // llvm::dbgs() << "\n Working on a designateOp:"
    //     << "\n\tsliceStartIdxVal: "  << getI64Val(sliceStartIdxVal)
    //     << "\n\tsliceEndIdxVal: "  << getI64Val(sliceLimitIdxVal)
    //     << "\n\tsliceStrideVal: "  << getI64Val(sliceStrideVal)
    //     << "\n";

    std::reverse(sliceStartIdxVals.begin(), sliceStartIdxVals.end());
    std::reverse(sliceLimitIdxVals.begin(), sliceLimitIdxVals.end());
    std::reverse(sliceStrideIdxVals.begin(), sliceStrideIdxVals.end());

    assert(tracking.valueMap.contains(memRef) &&
           "memRef should have corresponding value in StablehlO function!");
    auto memRefStablehlo = tracking.valueMap.lookup(memRef);
    assert(memRefStablehlo && "memRefStablehlo not exist!");
    auto stablehloSliceOp = stablehlo::SliceOp::create(
        opBuilder, funcOp.getLoc(),
        toCorrespondingTensorTy(resultOperand.getType()), memRefStablehlo,
        sliceStartIdxVals, sliceLimitIdxVals, sliceStrideIdxVals);
    tracking.valueMap.map(resultOperand, stablehloSliceOp.getResult());
  }
  return;
}

static void handleAssignOp(TrackingInfo &tracking, OpBuilder &opBuilder,
                           func::FuncOp funcOp, hlfir::AssignOp assignOp) {
  // example: hlfir.assign %155 to %150#0
  // Assign A to B
  assert(assignOp.getNumOperands() == 2 &&
         "Fail to assert assignOp has 2 Operands!");
  auto RHS = assignOp.getOperand(0); // assign from
  auto LHS = assignOp.getOperand(1); // assign to

  // 1. Simple case: naive assignment
  //
  // Example
  //  def (arg0, arg1):
  //    A = hlfir.declare arg0 // now arg0 is tracking A
  //    B = hlfir.declare arg1 // now arg1 is tracking B
  //    hlfir.assign A -> B       // now arg1 who was tracking B should also be
  //    tracking A, and value of A is arg0 return;
  //
  // Functional logic:
  //  \arg0 arg1 -> (arg0, arg0)
  //
  // 2. Slice case:
  //
  // Example
  //  def (arg0, arg1):
  //    A = hlfir.designate arg0[1:5:2]
  //    B = hlfir.designate arg1[1:5:2]
  //    hlfir.assign A -> B
  //    return;
  //
  // Functional logic:
  //  \arg0 arg1 ->
  //    A = arg0.slicing[1:5:2] // is A == arg0 ? in this case no.
  //    B = arg1.slicing[1:5:2] // is B == arg1 ? in this case no.
  //    newArg1 = create_a_partial_updated_B(arg1, arg0, B, A)
  //    return (arg0, newArg1)
  //
  //  In StableHLO, `scatter` is the operation for the above
  //  `create_a_partial_updatedB` logic
  //

  // TODO: generate scatter
  assert(tracking.valueMap.contains(LHS) && "WriteToVal is not tracked!");
  auto LHSStablehloVal = tracking.valueMap.lookup(LHS);
  assert(LHSStablehloVal && "LHSStablehloVal not exist!");
  // Deciding if we're slicing the LHS
  if (LHSStablehloVal.getDefiningOp() &&
      llvm::isa<stablehlo::SliceOp>(LHSStablehloVal.getDefiningOp())) {
    auto LHSStablehloSliceVal =
        llvm::dyn_cast<stablehlo::SliceOp>(LHSStablehloVal.getDefiningOp());
    assert(LHSStablehloSliceVal.getStrides()[0] == 1 &&
           "Only deal with continuous slice for now!");
    // in continuous case, we only need to patch once, so instead of:
    //
    // ```
    //  indices = stablehlo.iota(0: dimension) -> <count x i32>
    //  newWriteTo = stablehlo.scatter(writeTo, indices, readFrom)
    // ```
    //
    // We can have:
    // ```
    //  updates = RHS (it can be a slice like `LHS(1:5) = RHS(1:5)` or not a
    //  slice like `LHS(1:5) = RHS`) c0 = stablehlo.constant
    //  (LHSSlice->startIdx) indices = stablehlo.broadcast_in_dims [] on c0 ->
    //  <1xi64> // where to start update, A.K.A. scatter indices
    //  stablehlo.scatter(LHSOriginalNonSliceVal, indices, updates)
    //    {
    //      update_window_dims = [0] -> which dimension of the updates,
    //      scatter_dims_to_operand_dims = [0] -> scatter indices to
    //    }
    // ```
    //
    // In general, here is the rule:
    // y[ya_0:yb_0, ..., ya_n-1: yb_n-1] = x
    // x is already sliced, so we don't care about its starting and limiting idx
    // Here's how we set indices, and scatter_dims_to_operand_dims.
    // ```
    // for (int i = 0; i < n; i++) {
    //  update_window_dims.push_back(i); // if updating this dimension, we need
    //  this if ((yb_i - ya_i) < y.dim[i].size()) {
    //    // just updating part
    //    indices.push_back(y_ai);
    //    scatter_dims_to_operand_dims(i);
    //  }
    // }
    //

    llvm::DenseSet<int> startIndicesSet;
    llvm::SmallVector<Value> updatesIndicesOfEachDim;
    llvm::DenseMap<int, Value> idxToBroadcastRes;
    llvm::SmallVector<int64_t> updateWindowDims;
    llvm::SmallVector<int64_t> scatterDimsToOperandDims;
    auto LHSUnderlineTensorShape = llvm::dyn_cast<mlir::ShapedType>(
        LHSStablehloSliceVal.getOperand().getType());
    int updateDims = LHSStablehloSliceVal.getStartIndices().size();
    assert(LHSUnderlineTensorShape.getRank() == updateDims &&
           "UnderlineTensor should have rank the same with updateDims!");
    for (int i = 0; i < updateDims; i++) {
      auto sliceStartIdx = LHSStablehloSliceVal.getStartIndices()[i];
      auto sliceLimitIdx = LHSStablehloSliceVal.getLimitIndices()[i];
      updateWindowDims.push_back(i);

      if (sliceStartIdx > 0 || (sliceLimitIdx - sliceStartIdx) <
                                   LHSUnderlineTensorShape.getDimSize(i)) {
        // partial update in this dimension
        //
        if (!startIndicesSet.contains(sliceStartIdx)) {
          auto constOp = stablehlo::ConstantOp::create(
              opBuilder, funcOp.getLoc(),
              DenseElementsAttr::get(
                  RankedTensorType::get({}, opBuilder.getI64Type()),
                  sliceStartIdx));

          auto broadCastOp = stablehlo::BroadcastInDimOp::create(
              opBuilder, funcOp.getLoc(),
              RankedTensorType::get(
                  {1}, opBuilder.getI64Type()), // from <i64> to <1xi64>
              constOp.getResult(), opBuilder.getDenseI64ArrayAttr({}));

          startIndicesSet.insert(sliceStartIdx);
          idxToBroadcastRes[sliceStartIdx] = broadCastOp.getResult();
        }
        updatesIndicesOfEachDim.push_back(idxToBroadcastRes.at(sliceStartIdx));
        scatterDimsToOperandDims.push_back(i);
      }
    }

    // LHS is slicing of full size, do not need to use scatter logic!
    if (updatesIndicesOfEachDim.empty()) {
      auto RHSStablehloVal = tracking.valueMap.lookup(RHS);
      assert(RHSStablehloVal && "RHS is not in valueMap!\n");

      auto defOp = llvm::dyn_cast<hlfir::DesignateOp>(LHS.getDefiningOp());
      assert(defOp && "LHS should be defined by designateOp!");

      tracking.valueMap.map(LHS, RHSStablehloVal);
      tracking.valueMap.map(defOp.getMemref(), RHSStablehloVal);
      return;
    }

    Value scatterIndice;
    if (scatterDimsToOperandDims.size() == 1) {
      assert(startIndicesSet.size() == 1 ||
             (llvm::dbgs() << "Should be smaller or equal to "
                              "scatterDimsToOperandDims size: "
                           << startIndicesSet.size() << "\n",
              false));
      scatterIndice = updatesIndicesOfEachDim[0];
    } else {
      auto concatOp = stablehlo::ConcatenateOp::create(
          opBuilder, funcOp.getLoc(),
          mlir::RankedTensorType::get(
              updatesIndicesOfEachDim.size(),
              toCorrespondingTensorTy(updatesIndicesOfEachDim[0].getType())
                  .getElementType()), // mlir::Type resultType0
          updatesIndicesOfEachDim,
          opBuilder.getI64IntegerAttr(0)); // concat in dimension 0
      scatterIndice = concatOp.getResult();
    }

    auto RHSStablehloVal = tracking.valueMap.lookup(RHS);
    assert(RHSStablehloVal && ":( RHS of assignOP is not in valueMap, you have "
                              "to put it in the valuemap!\n");

    // static ScatterDimensionNumbersAttr get(::mlir::MLIRContext *context,
    // ::llvm::ArrayRef<int64_t> updateWindowDims, ::llvm::ArrayRef<int64_t>
    // insertedWindowDims, ::llvm::ArrayRef<int64_t> inputBatchingDims,
    // ::llvm::ArrayRef<int64_t> scatterIndicesBatchingDims,
    // ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims, int64_t
    // indexVectorDim);

    auto scatterDimNums = stablehlo::ScatterDimensionNumbersAttr::get(
        /*context*/ funcOp.getContext(),
        /*updateWindowDims*/ updateWindowDims,
        /*insertedWindowDims*/ {},
        /*inputBatchingDims*/ {},
        /*scatterIndicesBatchingDims*/ {},
        /*scatterDimsToOperandDims*/ scatterDimsToOperandDims,
        /*indexVectorDim*/ 0);

    auto scatterOp = stablehlo::ScatterOp::create(
        opBuilder, funcOp.getLoc(),
        LHSStablehloSliceVal.getOperand().getType(), // result Type
        LHSStablehloSliceVal
            .getOperand(), // refer to the original array, which is the input
        scatterIndice, RHSStablehloVal,
        /*scatter_dimension_numbers=*/scatterDimNums,
        /*indices_are_sorted*/
        BoolAttr::get(
            funcOp.getContext(),
            true), // Following 2 are set to true because of referencing JAX
                   // generated, can be fixed after knowing more information
        /*unique_indices*/ BoolAttr::get(funcOp.getContext(), true));

    assert(
        scatterOp.getNumResults() == 1 &&
        ":( I was thinking scatterOp should have one result here, but more?");

    // Insert computation block, in our case, just return the second one, which
    // is the new created
    auto storedBlock = opBuilder.getBlock();
    auto storedInsertPoint = opBuilder.getInsertionPoint();

    auto &computeRegion = scatterOp.getUpdateComputation();
    auto computaeBlock = opBuilder.createBlock(&computeRegion);
    auto elementTy = toCorrespondingTensorTy(
        LHSStablehloSliceVal.getType().getElementType());
    computaeBlock->addArgument(elementTy, funcOp.getLoc());
    computaeBlock->addArgument(elementTy, funcOp.getLoc());
    stablehlo::ReturnOp::create(opBuilder, funcOp.getLoc(),
                                {computaeBlock->getArgument(1)});

    opBuilder.setInsertionPoint(storedBlock, storedInsertPoint);

    // for args who is tracking the LHS's defining memref, it should now track
    // scatterOp's result

    auto defOp = llvm::dyn_cast<hlfir::DesignateOp>(LHS.getDefiningOp());
    assert(tracking.valueMap.contains(defOp.getMemref()) &&
           "defOp's memref should be in valueMap!");
    assert(tracking.argsTrackingMap.contains(
               tracking.valueMap.lookup(defOp.getMemref())) &&
           "Can find a tracking from arg!");

    // when using the LHS later, need to use corresponding value of RHS
    assert(tracking.valueMap.contains(RHS) && "RHS not exist!");
    tracking.valueMap.map(LHS, tracking.valueMap.lookup(RHS));

    tracking.valueMap.map(defOp.getMemref(), scatterOp.getResult(0));
  } else {
    // Old logic here when assignOp is to whole array, may also need to fix
    // value: B should tracking the same value as A
    assert(tracking.valueMap.contains(RHS) && "RHS not exist!");
    tracking.valueMap.map(LHS, tracking.valueMap.lookup(RHS));

    // argsTrackingMap: the arg which is tracking B now should tracking A?
    assert(tracking.valueMap.contains(LHS) && "LHS not exist!");
    tracking.argsTrackingMap.map(tracking.valueMap.lookup(LHS), RHS);
  }
  return;
}

static void handleFuncCallOp(
  TrackingInfo &tracking, 
  OpBuilder &opBuilder,
  func::FuncOp funcOp,
  func::CallOp callOp
) {
  auto paramsCount = callOp.getNumOperands();
  auto originalArgs = callOp.getArgOperands();
  llvm::SmallVector<Value> translatedArgs(paramsCount);
  for (int i = 0; i < paramsCount; i++) {
    Value originalArg = originalArgs[i];
    if (tracking.valueMap.contains(originalArg)) {
      translatedArgs[i] = tracking.valueMap.lookup(originalArg); 
    } else {
      llvm::errs() << "[JForce ERROR] Untracked call operand #" << i << ": ";
      originalArg.print(llvm::errs());
      llvm::errs() << "\nOriginal call op:\n";
      callOp.print(llvm::errs());
      llvm::errs() << "\n";
      llvm_unreachable("Untracked call operand in WorkdistributeToStableHLOPass");
    }
  }
  auto moduleOp = callOp->getParentOfType<ModuleOp>();
  assert(moduleOp);
  func::FuncOp calleeFunc = moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
  assert(calleeFunc);
  auto translatedCallOp = func::CallOp::create(opBuilder, callOp.getLoc(), calleeFunc, translatedArgs);
  assert(callOp.getNumResults() == translatedCallOp.getNumResults());
  updateTracking<OperationType::VAL_TO_VAL>(tracking, callOp.getOperands(), callOp.getResults(), translatedCallOp.getResults());
};

// TODO: this category is not correct, should rewrite
static void handleGeneralRelayOp(
  TrackingInfo &tracking, 
  OpBuilder &opBuilder,
  func::FuncOp funcOp,
  Operation* op
) {
  TypeSwitch<Operation*>(op)
    .Case<fir::DeclareOp>([&](fir::DeclareOp dop){
      auto declaredOprand = dop.getOperand(0);
      updateTracking<OperationType::MEM_TO_MEM>(tracking, {declaredOprand}, {dop.getResult()}, {});
    })
    .Case<fir::LoadOp>([&](fir::LoadOp loadOp) {
      updateTracking<OperationType::READ_VAL_FROM_MEM>(tracking, {loadOp.getMemref()}, {loadOp.getResult()}, {});
    })
    .Case<affine::AffineLoadOp>([&](affine::AffineLoadOp loadOp) {
      // TODO: handle more cases!
      assert(loadOp.getNumOperands() == 1);
      assert(loadOp.getIndices().empty());
      updateTracking<OperationType::READ_VAL_FROM_MEM>(tracking, {loadOp.getMemRef()}, {loadOp.getResult()}, {});
    })
    .Case<bufferization::ToTensorOp>([&](bufferization::ToTensorOp toTensorOp){
      updateTracking<OperationType::READ_VAL_FROM_MEM>(tracking, {toTensorOp.getOperand()}, {toTensorOp.getResult()}, {});
    })
    .Case<bufferization::ToBufferOp>([&](bufferization::ToBufferOp toBufferOp){
      updateTracking<OperationType::CREATE_MEM>(tracking, {}, {toBufferOp.getResult()}, {});
      updateTracking<OperationType::WRITE_VAL_TO_MEM>(tracking, {toBufferOp.getOperand()}, {toBufferOp.getResult()}, {});
    })
    .Case<bufferization::MaterializeInDestinationOp>([&](bufferization::MaterializeInDestinationOp mop){
      updateTracking<OperationType::WRITE_VAL_TO_MEM>(tracking, {mop.getSource()}, {mop.getDest()}, {});
    })
  ;
}

static void handleCreateValOps(
  TrackingInfo &tracking, 
  OpBuilder &opBuilder,
  func::FuncOp funcOp,
  Operation* op
) {
  TypeSwitch<Operation*>(op).
    Case<memref::AllocaOp>([&](memref::AllocaOp allocaOp){
      updateTracking<OperationType::CREATE_MEM>(tracking, {}, {allocaOp.getResult()}, {});
    });
};

static void handleStoreOps(
  TrackingInfo &tracking, 
  OpBuilder &opBuilder,
  func::FuncOp funcOp,
  Operation* op
) {
  TypeSwitch<Operation*>(op)
    .Case<affine::AffineStoreOp>([&](affine::AffineStoreOp storeOp){
      assert(storeOp.getIndices().empty());
      auto storeFrom = storeOp.getValueToStore(); 
      auto storeTo = storeOp.getMemRef();
      updateTracking<OperationType::WRITE_VAL_TO_MEM>(tracking, {storeFrom}, {storeTo}, {});
    })
    .Case<memref::StoreOp>([&](memref::StoreOp storeOp){
      assert(storeOp.getIndices().empty());
      auto storeFrom = storeOp.getValueToStore(); 
      auto storeTo = storeOp.getMemRef();
      updateTracking<OperationType::WRITE_VAL_TO_MEM>(tracking, {storeFrom}, {storeTo}, {});
    })
  ;
}


static void scanOperationsAndInserts(
    TrackingInfo &tracking, 
    OpBuilder &opBuilder,
    func::FuncOp hloFuncOp,
    Operation *op,
    const llvm::DenseMap<Value, llvm::SmallVector<int64_t>> &sliceShiftMap
) {
  DEBUG_PRINT_OP(op);
  llvm::TypeSwitch<Operation *>(op)
      .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
        stablehlo::ConstantOp stablehloConstOp;
        if (constOp.getType().isIndex()) {
          auto indexAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValueAttr());
          auto intAttr = IntegerAttr::get(IntegerType::get(opBuilder.getContext(), 64), indexAttr.getValue());
          stablehloConstOp = stablehlo::ConstantOp::create(opBuilder, hloFuncOp.getLoc(), intAttr);
        } else {
          stablehloConstOp = stablehlo::ConstantOp::create(opBuilder, hloFuncOp.getLoc(), constOp.getValueAttr());
        }
        updateTracking<CREATE_VAL>(tracking, {}, {constOp.getResult()}, {stablehloConstOp.getResult()});
      })
      .Case<fir::ZeroOp>([&](fir::ZeroOp zeroOp){
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
        auto stablehloZeroOp = stablehlo::ConstantOp::create(opBuilder, hloFuncOp.getLoc(), zeroAttr);
        updateTracking<CREATE_VAL>(tracking, {}, {zeroOp.getResult()}, {stablehloZeroOp.getResult()});
      })
      .Case<fir::AllocaOp, memref::AllocaOp>([&](auto allocaOp) {
        updateTracking<CREATE_MEM>(tracking, {}, {allocaOp.getResult()}, {});
      })
      .Case<fir::ConvertOp>([&] (fir::ConvertOp convertOp) {
        handleConvertOp(tracking, opBuilder, hloFuncOp, convertOp); 
      })
      .Case<func::CallOp>([&](func::CallOp callOp){
        handleFuncCallOp(tracking, opBuilder, hloFuncOp, callOp); 
      })
      // TODO: rewrite this case.
      .Case<
        fir::DeclareOp, fir::LoadOp, affine::AffineLoadOp,
        bufferization::ToTensorOp, bufferization::ToBufferOp,
        bufferization::MaterializeInDestinationOp
          >([&](Operation* op){
        handleGeneralRelayOp(tracking, opBuilder, hloFuncOp, op);
      })

      .Case<hlfir::YieldElementOp>([&](hlfir::YieldElementOp yeOp) {
        // Should find the corresponding the elementalOp and establish the
        // mapping between the yield value and the result of elementalOp
        assert(yeOp->getNumOperands() == 1 &&
               "Fail to assert YeOP has 1 operand!");
        auto yieldOperand = yeOp->getOperand(0);

        auto parentOp = yeOp.getParentOp();
        assert(llvm::isa<hlfir::ElementalOp>(parentOp) &&
               "Fail to assert the parentOP of YeOP is elementalOp!");
        auto parentOpResult = parentOp.getResult();

        auto stablehloArg = tracking.valueMap.lookup(yieldOperand);
        assert(stablehloArg && "stabelhloArg not exist!");
        tracking.valueMap.map(parentOpResult, stablehloArg);
        tracking.argsTrackingMap.map(stablehloArg, parentOpResult);
      })
      .Case<hlfir::AssignOp>([&](hlfir::AssignOp assignOp) {
        handleAssignOp(tracking, opBuilder, hloFuncOp, assignOp);
      })
      .Case<hlfir::DeclareOp>([&](hlfir::DeclareOp declareOp) {
        auto declaredOprand = declareOp.getOperand(0);
        for (unsigned int i = 0; i < declareOp->getNumResults(); i++) {
          assert(tracking.valueMap.contains(declaredOprand) && "declaredOperand not exist!");
          tracking.valueMap.map(declareOp->getOpResult(i),
                                tracking.valueMap.lookup(declaredOprand));
        }
      })
      .Case<hlfir::DesignateOp>([&](hlfir::DesignateOp designateOp) {
        handleDesignateOp(tracking, opBuilder, hloFuncOp, designateOp,
                          sliceShiftMap);
      })
      .Case<hlfir::ApplyOp>([&](hlfir::ApplyOp applyOp) {
        // example: %445 = "hlfir.apply"(%443, %arg9)
        auto resultOperand = applyOp->getOpResult(0);
        auto refArr = applyOp.getOperand(0);
        assert(tracking.valueMap.contains(refArr) && "refArr not exist!");
        tracking.valueMap.map(resultOperand, tracking.valueMap.lookup(refArr));
      })
      .Case<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, 
            arith::MulFOp, arith::MulIOp, arith::DivFOp, arith::DivSIOp,
            arith::CmpFOp, arith::CmpIOp>([&](auto arithBinaryOp) {
        handleArithBinaryOp(tracking, opBuilder, hloFuncOp, arithBinaryOp);
      })
      .Case<arith::SelectOp>([&](arith::SelectOp sop) {
        // %190 = "arith.select"(%189, %186, %187) : (i1, f64, f64) -> f64
        assert(sop.getNumOperands() == 3 && "Unexpected select oeprands size!");
        auto firCond = sop.getOperand(0);
        auto firOnTrue = sop.getOperand(1);
        auto firOnFalse = sop.getOperand(2);
        assert(tracking.valueMap.contains(firCond) &&
               "Should contain firCond!");
        assert(tracking.valueMap.contains(firOnTrue) &&
               "Should contain firOnTrue!");
        assert(tracking.valueMap.contains(firOnFalse) &&
               "Should contain firOnFalse!");

        // Consider example when comparing a tensor with a scalar 0: ReLU(x) =
        // MAX(x, 0) In such a case, we have to broadcast the operand!
        Value trueSrc = tracking.valueMap.lookup(firOnTrue);
        assert(trueSrc && "trueSrc not exist!");
        Value falseSrc = tracking.valueMap.lookup(firOnFalse);
        assert(falseSrc && "falseSrc not exist!");

        RankedTensorType trueTy =
            llvm::dyn_cast<RankedTensorType>(trueSrc.getType());
        RankedTensorType falseTy =
            llvm::dyn_cast<RankedTensorType>(falseSrc.getType());

        // StableHLO select requires true and false operands to have the same
        // shape. Insert BroadcastInDim if there is a scalar vs tensor mismatch.
        if (trueTy.getRank() != falseTy.getRank()) {
          Value largerOperand =
              (trueTy.getRank() > falseTy.getRank()) ? trueSrc : falseSrc;
          Value smallerOperand =
              (trueTy.getRank() > falseTy.getRank()) ? falseSrc : trueSrc;
          RankedTensorType targetTy =
              llvm::dyn_cast<RankedTensorType>(largerOperand.getType());

          DenseI64ArrayAttr diaa = opBuilder.getDenseI64ArrayAttr({});
          auto broadcastOp = stablehlo::BroadcastInDimOp::create(
              opBuilder, hloFuncOp.getLoc(), targetTy, smallerOperand, diaa);

          if (trueTy.getRank() > falseTy.getRank()) {
            falseSrc = broadcastOp.getResult();
          } else {
            trueSrc = broadcastOp.getResult();
          }
        }

        assert(tracking.valueMap.contains(firCond) && "firCond not exist!");
        auto stableHLOSelectRes = stablehlo::SelectOp::create(
            opBuilder, hloFuncOp.getLoc(), tracking.valueMap.lookup(firCond),
            trueSrc, falseSrc);
        tracking.valueMap.map(sop.getResult(), stableHLOSelectRes.getResult());
      })
      .Case<math::SinOp, math::ExpOp, math::SqrtOp, arith::NegFOp>([&](auto arithUnaryOp) {
            handleArithUnaryOp(tracking, opBuilder, hloFuncOp, arithUnaryOp);
      })
      .Case<hlfir::MatmulOp, hlfir::DotProductOp, hlfir::TransposeOp, hlfir::SumOp>([&](auto builtInOp) {
        handleBuiltinOperators(tracking, opBuilder, hloFuncOp, builtInOp);
      })
      .Case<hlfir::NoReassocOp>([&](hlfir::NoReassocOp nrop) {
        assert(tracking.valueMap.contains(nrop.getOperand()) &&
               "Operand of NoReassocOp is supposed to be in ValueMap!");
        // just ignore and pass to the result
        assert(tracking.valueMap.contains(nrop.getOperand()) && "nrop.getOperand not exist!");
        tracking.valueMap.map(nrop.getResult(),
                              tracking.valueMap.lookup(nrop.getOperand()));
      })
      
      .Case<affine::AffineStoreOp, memref::StoreOp>([&](Operation* op){
        handleStoreOps(tracking, opBuilder, hloFuncOp, op);
      })
      .Case<memref::AllocaOp>([&](Operation* op){
        handleCreateValOps(tracking, opBuilder, hloFuncOp, op);
      })
      // TODO: including other cases!
      .Default([](auto op) {
        if (!llvm::isa<func::FuncOp>(op)) {
          DEBUG_PRINT("Unhandled operation:");
          DEBUG_PRINT_OP(op);
        }
      });
}

static void terminateFunction(
  const TrackingInfo &tracking,
  OpBuilder &opBuilder, 
  func::FuncOp translatedFunc
) {
  auto argNum = translatedFunc.getNumArguments();
  mlir::SmallVector<Value> returnValues(argNum);
  for (unsigned int i = 0; i < argNum; i++) {
    auto currArg = translatedFunc.getArgument(i);
    assert(tracking.argsTrackingMap.contains(currArg) && "currArg not exist!");
    returnValues[i] = tracking.argsTrackingMap.lookup(currArg);
  }

  // Find the end of the last block.
  opBuilder.setInsertionPointToEnd(&translatedFunc.front());
  func::ReturnOp::create(opBuilder, translatedFunc.getLoc(), returnValues);
}

static bool isStableHLOFunction(::mlir::StringRef functionName) {
  auto pattern = llvm::formatv("^{0}[0-9]+_raised$", JIT_OUTLINE_AFFINE_FUNC_PREFIX).str(); 
  std::regex re(pattern);
  return std::regex_match(functionName.str(), re);
}

struct WorkdistributeToStableHLOPass
    : public PassWrapper<WorkdistributeToStableHLOPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WorkdistributeToStableHLOPass)

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
    registry.insert<mlir::bufferization::BufferizationDialect>();
  }

  StringRef getArgument() const override { 
    return "jforce-translate"; 
  }

  void runOnOperation() override {
    PROFILE_SCOPE("workdistributeToStableHLO", Phase::LOWERING_TO_STABLEHLO);
    auto moduleOp = getOperation(); 
    auto context = moduleOp.getContext();
    OpBuilder opBuilder(context);

    // Should not translate already StableHLO function.
    llvm::SmallVector<func::FuncOp>  funcsToReplace;
    moduleOp.walk([&](func::FuncOp fOp){
      if (!isStableHLOFunction(fOp.getName())) {
        funcsToReplace.push_back(fOp);
      }
    });

    for (auto oldFOp: funcsToReplace) {
      TrackingInfo trackingInfo;
      auto sliceShiftMap = extractSliceShifts(oldFOp);
      auto stableHLOFuncOp = createFunction(context, trackingInfo, oldFOp);
      opBuilder.setInsertionPointToStart(&stableHLOFuncOp.front());
      oldFOp->walk([&](Operation *op) {
        scanOperationsAndInserts(trackingInfo, opBuilder, stableHLOFuncOp, op, sliceShiftMap);
      });
      terminateFunction(trackingInfo, opBuilder, stableHLOFuncOp);
      opBuilder.setInsertionPointAfter(oldFOp);
      opBuilder.insert(stableHLOFuncOp);
      oldFOp.erase();
    }
  }
};
} // namespace

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createWorkdistributeToStableHLOPass() {
    return std::make_unique<WorkdistributeToStableHLOPass>();
  }

  void registerWorkdistributeToStableHLOPass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createWorkdistributeToStableHLOPass();});
  };
}
