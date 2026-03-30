#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "jit-manager.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "profiler.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockSupport.h>
#include "mlir/IR/Builders.h"
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <numeric>
#include <string>
#include <unordered_map>
#include "utilities.h"

using namespace mlir;

/// valueMap, argsTrackingMaps are 2 maps we'll keep updating when scanning 
/// - valueMap: tracking the each operand of FIR pointing to the value of each operand in Stablehlo function
/// - argsTrackingMap: tracking the current value of arguments of stablehlo pointing to, practically a reverse map of `valueMap`
struct TrackingInfo {
public:
  // Key: value in FIR function 
  // Value: value in StableHLO function
  mlir::IRMapping valueMap;
  // Key: value of one of StableHLO function's arguments 
  // Value: value in FIR function
  mlir::IRMapping argsTrackingMap;

  // Key: unique name
  // Value: corresponding StableHLO Value the definition mapping to   
  // Purpose: for tracking private variables, they are temporary variables, and should be dropped in later transformation
  mlir::DenseMap<mlir::StringAttr, mlir::Value> uniqueNamesMap;
};

///  Example of source type:
///  "!fir.ref<!fir.array<10xf32>>": convert to "tensor<10xf32>"
///  "!fir.ref<f32>": convert to "tensor<f32>"
///  "!hlfir.expr<shape>: convert to tensor<shape>"
[[deprecated("Should only be used when generating stablehlo op, and should reverse the dimensions")]]
static RankedTensorType convertBufferTyToTensorTy(mlir::Type srcTy) {
  // If it's already a tensor type, then no need to convert
  if (llvm::isa<RankedTensorType>(srcTy)) {
    return llvm::dyn_cast<RankedTensorType>(srcTy);
  }


  return llvm::TypeSwitch<mlir::Type, RankedTensorType>(srcTy)
  .Case<hlfir::ExprType>([](hlfir::ExprType expTy){
    return RankedTensorType::get(expTy.getShape(), expTy.getEleTy());
  })
  .Case<fir::BoxType>([](fir::BoxType bTy){
    return convertBufferTyToTensorTy(bTy.getEleTy());
  })
  .Case<fir::ReferenceType>([](fir::ReferenceType refTy){
    return convertBufferTyToTensorTy(refTy.getEleTy());
  })
  .Case<fir::SequenceType>([](fir::SequenceType seqTy){
    return RankedTensorType::get(seqTy.getShape(), seqTy.getEleTy());
  })
  .Default([&](auto scTy){
    // Suppose this is a scalar type
    return RankedTensorType::get({}, scTy);
  });
}

///  Example of source type:
///  "!fir.ref<!fir.array<10xf32>>": convert to "tensor<10xf32>"
///  "!fir.ref<!fir.array<10x20xf32>>": convert to "tensor<20x10xf32>", notice it is reversed
///  "!fir.ref<f32>": convert to "tensor<f32>"
static RankedTensorType toCorrespondingTensorTy(mlir::Type srcTy) {
  // If it's already a tensor type, then no need to convert
  if (llvm::isa<RankedTensorType>(srcTy)) {
    return llvm::dyn_cast<RankedTensorType>(srcTy);
  }

  return llvm::TypeSwitch<mlir::Type, RankedTensorType>(srcTy)
  .Case<hlfir::ExprType>([](hlfir::ExprType expTy){
    auto shape = llvm::to_vector(expTy.getShape());
    std::reverse(shape.begin(), shape.end());
    return RankedTensorType::get(shape, expTy.getEleTy());
  })
  .Case<fir::BoxType>([](fir::BoxType bTy){
    return toCorrespondingTensorTy(bTy.getEleTy());
  })
  .Case<fir::ReferenceType>([](fir::ReferenceType refTy){
    return toCorrespondingTensorTy(refTy.getEleTy());
  })
  .Case<fir::SequenceType>([](fir::SequenceType seqTy){
    auto shape = llvm::to_vector(seqTy.getShape());
    std::reverse(shape.begin(), shape.end());
    return RankedTensorType::get(shape, seqTy.getEleTy());
  })
  .Default([&](auto scTy){
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
static func::FuncOp createFunction(mlir::MLIRContext* context,
                                   TrackingInfo &tracking,
                                   const func::FuncOp& inputOp) {
  auto &firstRegion = inputOp->getRegion(0);
  auto &block = firstRegion.getBlocks().front();

  mlir::SmallVector<Type> argsTypes;
  for (const auto& arg : block.getArguments()) {
    argsTypes.push_back(toCorrespondingTensorTy(arg.getType()));
  }

  auto funcType = mlir::FunctionType::get(context, argsTypes, argsTypes);
  auto funcOp =
      func::FuncOp::create(inputOp->getLoc(), "main", funcType, {});
  // we need to update the valueMap!
  funcOp.addEntryBlock();

  for (int i = 0; i < block.getNumArguments(); i++) {
    Value firFuncArg = block.getArgument(i);
    Value stablehloFuncArg = funcOp.getArgument(i);
    tracking.valueMap.map(firFuncArg, stablehloFuncArg); 
    tracking.argsTrackingMap.map(stablehloFuncArg, firFuncArg);
  }
  return funcOp;
}


static void handleArithUnaryOp(
  TrackingInfo& tracking, 
  OpBuilder& opBuilder, 
  func::FuncOp& funcOp, 
  Operation* unaryOp
) {
  assert(unaryOp->getNumResults() == 1 && unaryOp->getNumOperands() == 1);
  auto operand = unaryOp->getOperand(0);
  auto result = unaryOp->getResult(0);
  auto operandSrc = tracking.valueMap.lookup(operand);
  assert(operandSrc && "OperandSource should exist!");
  llvm::TypeSwitch<Operation*>(unaryOp)
    .Case([&](mlir::math::SinOp sop){
      // should have same type
      auto resTy = toCorrespondingTensorTy(operandSrc.getType());   
      auto sinOp = stablehlo::SineOp::create(opBuilder, funcOp.getLoc(), resTy, operandSrc, {});
      tracking.valueMap.map(result, sinOp.getResult());       
    })
    .Case([&](mlir::math::ExpOp eop){
      auto resTy = toCorrespondingTensorTy(operandSrc.getType());
      auto stablehloExOp = stablehlo::ExpOp::create(opBuilder, funcOp.getLoc(), resTy, operandSrc, {});
      tracking.valueMap.map(result, stablehloExOp.getResult());
    });
}

/// Only support increase one dimension right now, for example:
/// - tensor<f32> -> tensor<10xf32>
/// - tensor<10xf32> -> tensor<10x10xf32>
/// 
/// Ref: https://openxla.org/stablehlo/spec#broadcast_in_dim
static void handleArithBinaryOp(TrackingInfo& tracking, 
                                OpBuilder &opBuilder, 
                                func::FuncOp& funcOp, 
                                Operation* arithOp) {
  // llvm::dbgs() << "\n Handling Arith Binary OP: " << getMLIROperationAsString(arithOp) << "\n";

  assert(arithOp->getNumOperands() == 2 && arithOp->getNumResults() == 1);
  Value operand1 = arithOp->getOperand(0);
  Value operand2 = arithOp->getOperand(1);
  Value result = arithOp->getResult(0);

  assert(tracking.valueMap.contains(operand1) && "ValueMap supposed to contain operand1!");
  assert(tracking.valueMap.contains(operand2) && "ValueMap supposed to contain operand2!");
  Value operand1Src = tracking.valueMap.lookup(operand1);
  Value operand2Src = tracking.valueMap.lookup(operand2);


  RankedTensorType o1Type = toCorrespondingTensorTy(operand1Src.getType()); 
  RankedTensorType o2Type = toCorrespondingTensorTy(operand2Src.getType()); 
  assert(o1Type.hasRank() && o2Type.hasRank());

  Value largerOperand, smallerOperand; 
  if (o1Type.getRank() >= o2Type.getRank()) {
    largerOperand = operand1Src;
    smallerOperand = operand2Src;
  } else {
    largerOperand = operand2Src;
    smallerOperand = operand1Src;
  }
  RankedTensorType targetType = toCorrespondingTensorTy(largerOperand.getType());

  // insert the broadcast
  if (o1Type.getRank() != o2Type.getRank()) {
    DenseI64ArrayAttr diaa = opBuilder.getDenseI64ArrayAttr({});
    auto broadcastInDimOp = stablehlo::BroadcastInDimOp::create(opBuilder, funcOp.getLoc(), targetType, smallerOperand, diaa);
    smallerOperand = broadcastInDimOp.getResult();
  }

  // insert the arith operation
  Value stablehloRes;
  llvm::TypeSwitch<Operation *>(arithOp)
    .Case<arith::AddFOp>([&](arith::AddFOp addOp){
      auto stablehloAddOp = stablehlo::AddOp::create(
        opBuilder, 
        funcOp.getLoc(), 
        targetType, 
        largerOperand,
        smallerOperand
      );
      stablehloRes = stablehloAddOp.getResult();
    })
    .Case<arith::MulFOp>([&](arith::MulFOp){
      auto stablehloMulOp = stablehlo::MulOp::create(
        opBuilder, 
        funcOp.getLoc(), 
        targetType, 
        largerOperand, 
        smallerOperand
      );
      stablehloRes = stablehloMulOp.getResult();
    })
    // Can not exchange
    .Case<arith::SubFOp>([&](arith::SubFOp subOp){
      if (o1Type.getRank() >= o2Type.getRank()){
        stablehloRes = stablehlo::SubtractOp::create(
          opBuilder,
          funcOp.getLoc(),
          targetType,
          largerOperand,
          smallerOperand
        ).getResult();
      } else {
        stablehloRes = stablehlo::SubtractOp::create(
          opBuilder,
          funcOp.getLoc(),
          targetType,
          smallerOperand,
          largerOperand
        ).getResult();
      }
    })
    .Case<arith::DivFOp>([&](arith::DivFOp){
      if (o1Type.getRank() >= o2Type.getRank()) {
        stablehloRes = stablehlo::DivOp::create(
          opBuilder,
          funcOp.getLoc(),
          targetType,
          largerOperand,
          smallerOperand
        ).getResult();
      } else {
        stablehloRes = stablehlo::DivOp::create(
          opBuilder,
          funcOp.getLoc(),
          targetType,
          smallerOperand,
          largerOperand
        ).getResult();
      }
    })
    .Default([](auto){
      llvm::errs() << "Unknown arith operation! \n";
      return;
    });

  tracking.valueMap.map(result, stablehloRes);
  return;
}

static void handleBuiltinOperators(TrackingInfo& tracking,
                                   OpBuilder& opBuilder,
                                   func::FuncOp& funcOp,
                                   Operation* op) {
  llvm::TypeSwitch<Operation*>(op)
    .Case<hlfir::DotProductOp>([&](hlfir::DotProductOp dpOp){
      // example: %151 = hlfir.dot_product %148#0 %150#0 
      // TODO: only support 1d arrays dot product
      assert(dpOp.getNumOperands() == 2 && "Failure of expecting dot product op has 2 operands!\n");
      auto lhs = tracking.valueMap.lookup(dpOp.getOperand(0));
      auto rhs = tracking.valueMap.lookup(dpOp.getOperand(1));
      
      auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
      if (!lhsTy) {
        llvm::errs() << "Unexpected lhs type!\n";
        exit(1);
      }
      auto scalarType = RankedTensorType::get({}, lhsTy.getElementType());
      stablehlo::DotDimensionNumbersAttr attr = stablehlo::DotDimensionNumbersAttr::get(funcOp.getContext(), {}, {}, {0}, {0});
      ArrayAttr precisionConfig = {};
      stablehlo::DotAlgorithmAttr algoAttr = {};
      auto stablehloDotProductOp = stablehlo::DotGeneralOp::create(opBuilder, funcOp->getLoc(), scalarType, lhs, rhs, attr, precisionConfig, algoAttr);
      tracking.valueMap.map(dpOp.getResult(), stablehloDotProductOp.getResult());
    })
    .Case<hlfir::MatmulOp>([&](hlfir::MatmulOp mmOp){
      // %36 = hlfir.matmul %33#0 %35#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>) -> !hlfir.expr<?x?xf64>
      auto op0 = tracking.valueMap.lookup(mmOp.getOperand(0));
      auto op1 = tracking.valueMap.lookup(mmOp.getOperand(1));
      auto op0Ty = toCorrespondingTensorTy(op0.getType());
      auto op1Ty = toCorrespondingTensorTy(op1.getType());
      auto resTy = RankedTensorType::get(
        llvm::SmallVector<int64_t>{op0Ty.getShape()[0], op1Ty.getShape()[1]}, 
        op0Ty.getElementType()
      );

      mlir::ArrayAttr config = {};
      mlir::stablehlo::DotAlgorithmAttr algo = {};
      auto dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        funcOp.getContext(),
        SmallVector<int64_t> {},
        SmallVector<int64_t> {},
        SmallVector<int64_t> {1},
        SmallVector<int64_t> {0});

      auto stablehloDotGeneralOp = stablehlo::DotGeneralOp::create(
        opBuilder,
        funcOp.getLoc(),
        resTy,
        op1,
        op0,
        dims,
        config,
        algo
      );
      tracking.valueMap.map(mmOp.getResult(), stablehloDotGeneralOp.getResult());
    })
    .Case<hlfir::TransposeOp>([&](hlfir::TransposeOp tOp){
      // %24 = hlfir.transpose %23#0 : (!fir.box<!fir.array<?x?xf64>>) -> !hlfir.expr<?x?xf64>
      auto op = tracking.valueMap.lookup(tOp.getOperand());
      auto opTy = toCorrespondingTensorTy(op.getType());
      auto resTy = RankedTensorType::get(
        llvm::SmallVector<int64_t>{opTy.getShape()[1], opTy.getShape()[0]}, 
        opTy.getElementType()
      );
      auto stablehloTransposeOp = stablehlo::TransposeOp::create(
        opBuilder, 
        funcOp.getLoc(),
        resTy,
        op,
        llvm::SmallVector<int64_t>{1, 0}
      );
      tracking.valueMap.map(tOp.getResult(), stablehloTransposeOp.getResult());
    })
    .Default([](auto){
      llvm::errs() << "Not Supported Builtin Operators!\n";
      std::exit(EXIT_FAILURE);
    });
}


/// DesignateOp can generate slice
static void handleDesignateOp(
  TrackingInfo& tracking, 
  OpBuilder &opBuilder, 
  func::FuncOp funcOp, 
  hlfir::DesignateOp designateOp, 
  const llvm::DenseMap<Value, llvm::SmallVector<int>>& sliceShiftMap
) {
  auto isTriplet = designateOp.getIsTriplet();
  auto resultOperand = designateOp.getResult();
  auto memRef = designateOp.getMemref();

  bool isSlicing = [&]() {
    auto containsSlicing = false;
    for (bool tri: isTriplet) {
      containsSlicing = tri || containsSlicing;
      if (containsSlicing) {
        return true;
      }
    }
    return containsSlicing;
  }();

  if (!isSlicing){
    // example: %451 = "hlfir.designate"(%447#0, %arg9) 
    // Often inside of an elemental operation
    tracking.valueMap.map(resultOperand, tracking.valueMap.lookup(memRef));
  } else {
    // example 1: 1D slicing %1#0 [%c1: %c5: %c1]
    // - %7 = hlfir.designate %1#0 (%c1:%c5:%c1)  shape %4 : (!fir.box<!fir.array<?xf64>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
    //
    // example 2: 2D slicing %6#0 []
    //- %8 = "hlfir.designate"(%6#0, %1, %2, %1, %1, %0, %1, %7) <{is_triplet = array<i1: true, true>, operandSegmentSizes = array<i32: 1, 0, 6, 0, 1, 0>}> : (!fir.box<!fir.array<5x5xf64>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<2x5xf64>>
    auto indices = designateOp.getIndices();    
    assert(indices.size() >= 3 && indices.size() % 3 == 0 && "Unexpected Indices!");
    auto rank = indices.size()/3;

    // index shifts
    auto defOp = llvm::dyn_cast<hlfir::DeclareOp>(designateOp.getMemref().getDefiningOp());
    assert(defOp && "Defining Op of designateOp memref should be a declareOp!");

    llvm::SmallVector<int> defaultShifts(rank, 1);
    const llvm::SmallVector<int>* shifts;
    auto ssOp = llvm::dyn_cast<fir::ShapeShiftOp>(defOp.getShape().getDefiningOp());
    if(ssOp) {
      auto it = sliceShiftMap.find(ssOp.getResult());
      assert(it != sliceShiftMap.end() && "All shapeshifts should already have been recorded!");
      shifts = &(it->getSecond());
    } else {
      shifts = &defaultShifts;
    }    
    assert((shifts->size() == rank) || 
      (llvm::errs() << "Shifts size should be the same with dimension size! While shifts size: " <<  shifts->size() << ", while rank =" << rank, false));

    
    llvm::SmallVector<int64_t> sliceStartIdxVals;
    llvm::SmallVector<int64_t> sliceLimitIdxVals;
    llvm::SmallVector<int64_t> sliceStrideIdxVals;

   auto getI64Val = [](const mlir::Value& idxVal) -> int64_t {
      arith::ConstantIndexOp constOp = llvm::dyn_cast<arith::ConstantIndexOp>(idxVal.getDefiningOp());
      if (!constOp) {
        std::cerr << "This is not constantOp, you stupid!\n"; 
        std::exit(EXIT_FAILURE);
      }
      return constOp.value();
    };


    // example, FortranSlice(-1, 2, 1) with shift -1, should be FortranSlice(1, 4, 1), should be stablehlo.slice(0, 4, 1)
    // example, FortranSlice(1, 2, 1) with shift 0, should be FortranSlice(1, 2, 1), should be stablehlo.slice(0, 3, 1)
    for (int i = 0; i < rank; i++) {
      auto offSet = i * 3;

      auto startIdxVal = getI64Val(indices[offSet]) - (*shifts)[i];   
      auto limitIdxVal = getI64Val(indices[offSet+1]) - (*shifts)[i] + 1; // In stablehlo, the limit idx is not included 
      assert(getI64Val(indices[offSet + 2]) == 1 && "Only dealing with stride = 1 for now!!!");
      auto strideIdxVal = getI64Val(indices[offSet + 2]); 

      sliceStartIdxVals.push_back(startIdxVal);
      sliceLimitIdxVals.push_back(limitIdxVal);
      sliceStrideIdxVals.push_back(strideIdxVal);
    }
       
    // create slice operation
    // we're not inserting in place, so donot set the insertion point of opBuilder
    
    // Debugging...
    // llvm::dbgs() << "\n Working on a designateOp:"
    //     << "\n\tsliceStartIdxVal: "  << getI64Val(sliceStartIdxVal)
    //     << "\n\tsliceEndIdxVal: "  << getI64Val(sliceLimitIdxVal)
    //     << "\n\tsliceStrideVal: "  << getI64Val(sliceStrideVal) 
    //     << "\n";
    
    std::reverse(sliceStartIdxVals.begin(),  sliceStartIdxVals.end()); 
    std::reverse(sliceLimitIdxVals.begin(),  sliceLimitIdxVals.end());
    std::reverse(sliceStrideIdxVals.begin(),  sliceStrideIdxVals.end());
    
    assert(tracking.valueMap.contains(memRef) && "memRef should have corresponding value in StablehlO function!");
    auto memRefStablehlo = tracking.valueMap.lookup(memRef);
    auto stablehloSliceOp = stablehlo::SliceOp::create(
      opBuilder, 
      funcOp.getLoc(),
      toCorrespondingTensorTy(resultOperand.getType()),
      memRefStablehlo,
      sliceStartIdxVals,
      sliceLimitIdxVals,
      sliceStrideIdxVals
    );
    tracking.valueMap.map(resultOperand, stablehloSliceOp.getResult());
  }
  return;
}


static void handleAssignOp(TrackingInfo& tracking, OpBuilder &opBuilder, func::FuncOp funcOp, hlfir::AssignOp assignOp) {
  // example: hlfir.assign %155 to %150#0 
  // Assign A to B
  assert(assignOp.getNumOperands() == 2 && "Fail to assert assignOp has 2 Operands!");
  auto RHS = assignOp.getOperand(0);          // assign from
  auto LHS = assignOp.getOperand(1);   // assign to

  // 1. Simple case: naive assignment
  // 
  // Example
  //  def (arg0, arg1):
  //    A = hlfir.declare arg0 // now arg0 is tracking A
  //    B = hlfir.declare arg1 // now arg1 is tracking B
  //    hlfir.assign A -> B       // now arg1 who was tracking B should also be tracking A, and value of A is arg0
  //    return;
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
  //  In StableHLO, `scatter` is the operation for the above `create_a_partial_updatedB` logic
  //

  // TODO: generate scatter 
  assert(tracking.valueMap.contains(LHS) && "WriteToVal is not tracked!");
  auto LHSStablehloVal = tracking.valueMap.lookup(LHS);
  // Deciding if we're slicing the LHS
  if (LHSStablehloVal.getDefiningOp() && llvm::isa<stablehlo::SliceOp>(LHSStablehloVal.getDefiningOp())) {
    auto LHSStablehloSliceVal = llvm::dyn_cast<stablehlo::SliceOp>(LHSStablehloVal.getDefiningOp());
    assert(LHSStablehloSliceVal.getStrides()[0] == 1 && "Only deal with continuous slice for now!");
    // in continuous case, we only need to patch once, so instead of:
    //
    // ```
    //  indices = stablehlo.iota(0: dimension) -> <count x i32> 
    //  newWriteTo = stablehlo.scatter(writeTo, indices, readFrom)
    // ```
    //
    // We can have:
    // ```
    //  updates = RHS (it can be a slice like `LHS(1:5) = RHS(1:5)` or not a slice like `LHS(1:5) = RHS`)
    //  c0 = stablehlo.constant (LHSSlice->startIdx)
    //  indices = stablehlo.broadcast_in_dims [] on c0 -> <1xi64> // where to start update, A.K.A. scatter indices
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
    //  update_window_dims.push_back(i); // if updating this dimension, we need this
    //  if ((yb_i - ya_i) < y.dim[i].size()) {
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
    auto LHSUnderlineTensorShape = llvm::dyn_cast<mlir::ShapedType>(LHSStablehloSliceVal.getOperand().getType());
    int updateDims = LHSStablehloSliceVal.getStartIndices().size();
    assert(LHSUnderlineTensorShape.getRank() == updateDims && "UnderlineTensor should have rank the same with updateDims!");
    for (int i = 0; i < updateDims; i++) {
      auto sliceStartIdx = LHSStablehloSliceVal.getStartIndices()[i];  
      auto sliceLimitIdx = LHSStablehloSliceVal.getLimitIndices()[i];
      updateWindowDims.push_back(i);
      
      if (sliceStartIdx > 0 || (sliceLimitIdx - sliceStartIdx) < LHSUnderlineTensorShape.getDimSize(i)) {
        // partial update in this dimension
        //
        if (!startIndicesSet.contains(sliceStartIdx)) {
          auto constOp = stablehlo::ConstantOp::create(
              opBuilder, 
              funcOp.getLoc(), 
              DenseElementsAttr::get(RankedTensorType::get({}, opBuilder.getI64Type()), sliceStartIdx));

          auto broadCastOp = stablehlo::BroadcastInDimOp::create(
              opBuilder,
              funcOp.getLoc(),
              RankedTensorType::get({1}, opBuilder.getI64Type()), // from <i64> to <1xi64>
              constOp.getResult(),
              opBuilder.getDenseI64ArrayAttr({}));

          startIndicesSet.insert(sliceStartIdx);
          idxToBroadcastRes[sliceStartIdx] = broadCastOp.getResult();
        }
        updatesIndicesOfEachDim.push_back(idxToBroadcastRes.at(sliceStartIdx)); 
        scatterDimsToOperandDims.push_back(i);    
      } 
    }

    Value scatterIndice;
    if (scatterDimsToOperandDims.size() == 1) {
      assert(startIndicesSet.size() == 1 || 
             (llvm::dbgs() << "Should be smaller or equal to scatterDimsToOperandDims size: " << startIndicesSet.size() << "\n", false));
      scatterIndice = updatesIndicesOfEachDim[0];
    } else {
      auto concatOp = stablehlo::ConcatenateOp::create(
        opBuilder, 
        funcOp.getLoc(), 
        mlir::RankedTensorType::get(
          updatesIndicesOfEachDim.size(), 
          toCorrespondingTensorTy(updatesIndicesOfEachDim[0].getType()).getElementType()),// mlir::Type resultType0 
        updatesIndicesOfEachDim,
        opBuilder.getI64IntegerAttr(0)); // concat in dimension 0
      scatterIndice = concatOp.getResult();
    }
    
    auto RHSStablehloVal = tracking.valueMap.lookup(RHS); 
    assert(RHSStablehloVal && ":( RHS of assignOP is not in valueMap, you have to put it in the valuemap!\n");

    // static ScatterDimensionNumbersAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> updateWindowDims, ::llvm::ArrayRef<int64_t> insertedWindowDims, ::llvm::ArrayRef<int64_t> inputBatchingDims, ::llvm::ArrayRef<int64_t> scatterIndicesBatchingDims, ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim);
     
    auto scatterDimNums = stablehlo::ScatterDimensionNumbersAttr::get(
      /*context*/ funcOp.getContext(),
      /*updateWindowDims*/ updateWindowDims,
      /*insertedWindowDims*/ {},
      /*inputBatchingDims*/ {},
      /*scatterIndicesBatchingDims*/ {},
      /*scatterDimsToOperandDims*/ scatterDimsToOperandDims, 
      /*indexVectorDim*/ 0
    );

    auto scatterOp = stablehlo::ScatterOp::create(
      opBuilder,
      funcOp.getLoc(),
      LHSStablehloSliceVal.getOperand().getType(), // result Type
      LHSStablehloSliceVal.getOperand(), // refer to the original array, which is the input
      scatterIndice,
      RHSStablehloVal,
      /*scatter_dimension_numbers=*/ scatterDimNums,
      /*indices_are_sorted*/ BoolAttr::get(funcOp.getContext(), true), // Following 2 are set to true because of referencing JAX generated, can be fixed after knowing more information
      /*unique_indices*/ BoolAttr::get(funcOp.getContext(), true)
    );

    // // scatter is very bug prone, this is for debugging
    // [&](){
    //   llvm::dbgs() << "\n\nDebugging Info for scatterOp: \n";
    //   llvm::dbgs() << "> Left side slice:\n";
    //   LHSStablehloSliceVal.print(llvm::dbgs());
    //   llvm::dbgs() << "\n> Right side slice:\n";
    //   auto RHSStablehloSliceVal = llvm::dyn_cast<stablehlo::SliceOp>(RHSStablehloVal.getDefiningOp());
    //   if (RHSStablehloSliceVal) {
    //     RHSStablehloSliceVal.print(llvm::dbgs());
    //   }
    //
    //   llvm::dbgs() << "\n Update Window Dims: ";
    //   llvm::interleaveComma(updateWindowDims, llvm::dbgs());
    //   llvm::dbgs() << "\n Scatter Dims To Operand Dims: ";
    //   llvm::interleaveComma(scatterDimsToOperandDims, llvm::dbgs());
    //
    //   auto indices = llvm::dyn_cast<stablehlo::ConcatenateOp>(scatterIndice.getDefiningOp());
    //   if (indices) {
    //     llvm::dbgs() << "\n ScatterIndices(update indices): \n";
    //     indices.print(llvm::dbgs());
    //   }
    // }(); 
    //
    assert(scatterOp.getNumResults() == 1 && ":( I was thinking scatterOp should have one result here, but more?");

    // Insert computation block, in our case, just return the second one, which is the new created
    auto storedBlock = opBuilder.getBlock();
    auto storedInsertPoint = opBuilder.getInsertionPoint();

    auto& computeRegion = scatterOp.getUpdateComputation();
    auto computaeBlock = opBuilder.createBlock(&computeRegion);
    auto elementTy = toCorrespondingTensorTy(LHSStablehloSliceVal.getType().getElementType()); 
    computaeBlock->addArgument(elementTy, funcOp.getLoc());
    computaeBlock->addArgument(elementTy, funcOp.getLoc());
    stablehlo::ReturnOp::create(opBuilder, funcOp.getLoc(), {computaeBlock->getArgument(1)});

    opBuilder.setInsertionPoint(storedBlock, storedInsertPoint);

    // for args who is tracking the LHS's defining memref, it should now track scatterOp's result
    
    auto defOp = llvm::dyn_cast<hlfir::DesignateOp>(LHS.getDefiningOp());
    assert(tracking.valueMap.contains(defOp.getMemref()) && "defOp's memref should be in valueMap!");
    assert(tracking.argsTrackingMap.contains(tracking.valueMap.lookup(defOp.getMemref())) && "Can find a tracking from arg!");
    
    // when using the LHS later, need to use corresponding value of RHS
    tracking.valueMap.map(LHS, tracking.valueMap.lookup(RHS));

    tracking.valueMap.map(defOp.getMemref(), scatterOp.getResult(0));
  } else {
    // Old logic here when assignOp is to whole array, may also need to fix
    // value: B should tracking the same value as A
    tracking.valueMap.map(LHS, tracking.valueMap.lookup(RHS));  

    // argsTrackingMap: the arg which is tracking B now should tracking A?
    tracking.argsTrackingMap.map(tracking.valueMap.lookup(LHS), RHS);
  }
  return;
}

static void scanOperationsAndInserts(
  TrackingInfo& tracking,
  OpBuilder &opBuilder, 
  func::FuncOp& funcOp, // TODO: do not need &
  Operation *op,
  const llvm::DenseMap<Value, llvm::SmallVector<int>>& sliceShiftMap,
  llvm::DenseSet<Value>& privateValSet
) {
  DEBUG_PRINT("Handling Op: " + getMLIROperationAsString(op));
  llvm::TypeSwitch<Operation *>(op)
      .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
        if (constOp.getResult().getType().isIndex()) {
          // This is just shape info, just return
          return;
        }
        auto stablehloConstOp = stablehlo::ConstantOp::create(opBuilder, funcOp.getLoc(), constOp.getValueAttr());
        tracking.valueMap.map(constOp.getResult(), stablehloConstOp.getResult());
      })
      .Case<hlfir::YieldElementOp>([&](hlfir::YieldElementOp yeOp){
        // Should find the corresponding the elementalOp and establish the mapping between the yield value and the result of elementalOp
        assert(yeOp->getNumOperands() == 1 && "Fail to assert YeOP has 1 operand!");
        auto yieldOperand = yeOp->getOperand(0); 

        auto parentOp = yeOp.getParentOp();
        assert(llvm::isa<hlfir::ElementalOp>(parentOp) && "Fail to assert the parentOP of YeOP is elementalOp!");
        auto parentOpResult = parentOp.getResult();

        auto stablehloArg = tracking.valueMap.lookup(yieldOperand);
        tracking.valueMap.map(parentOpResult, stablehloArg);
        tracking.argsTrackingMap.map(stablehloArg, parentOpResult);
      })
      .Case<hlfir::AssignOp>([&](hlfir::AssignOp assignOp) {
        handleAssignOp(tracking, opBuilder, funcOp, assignOp);
      })
      .Case<hlfir::DeclareOp>([&](hlfir::DeclareOp declareOp) {
        // To process private, we need to record the binded name to see if there's another one will shadow this
        auto uniqueName = declareOp.getUniqName();
        assert(uniqueName && "DeclareOp should have UniqueName!\n");
        if (tracking.uniqueNamesMap.contains(uniqueName)) {  
          // This is shadowing created by a private construct, thus in a teams
          // Example: 
          //    "omp.teams"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0>}> ({
          //    %5 = "fir.alloca"(%0) <{bindc_name = "z", in_type = !fir.array<?xf64>, operandSegmentSizes = array<i32: 0, 1>, pinned, uniq_name = "_QFFrun_benchmarkEz"}> : (index) -> !fir.ref<!fir.array<?xf64>>
          //    %6:2 = "hlfir.declare"(%5, %1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, storage_offset = 0 : ui64, uniq_name = "_QFFrun_benchmarkEz"}> : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<100xf64>>, !fir.ref<!fir.array<100xf64>>)
          assert(mlir::isa<omp::TeamsOp>(declareOp->getParentOp()) 
                 && "The declareOp because of private construct shoulded be contained in a TeamsOP!\n");
          auto defOpOfOperand = declareOp.getOperand(0).getDefiningOp();
          assert(mlir::isa<fir::AllocaOp>(defOpOfOperand)
                 && "The defining Op of the operand should be an fir::AllocaOp!\n");


          // declared result should be mapped as the original mapping   
          auto stableHLOValIt = tracking.uniqueNamesMap.find(uniqueName);
          assert(stableHLOValIt != tracking.uniqueNamesMap.end() 
                 && "Original FIR Val Arg should have set!\n");
          Value stableHLOVal = stableHLOValIt ->second;
          for (unsigned int i = 0; i < declareOp.getNumResults(); i++) {
            tracking.valueMap.map(declareOp->getOpResult(i), stableHLOVal);  
          } 
          tracking.argsTrackingMap.map(stableHLOVal, declareOp.getResult(0));
          privateValSet.insert(stableHLOVal);
        } else {
          // This is a new declaration
          auto declaredOprand = declareOp.getOperand(0);
          for (unsigned int i = 0; i < declareOp->getNumResults(); i++) {
            tracking.valueMap.map(declareOp->getOpResult(i), tracking.valueMap.lookup(declaredOprand));
          }
          tracking.argsTrackingMap.map(tracking.valueMap.lookup(declaredOprand), declareOp->getOpResult(0));
          tracking.uniqueNamesMap[uniqueName] = tracking.valueMap.lookup(declaredOprand);
        }
      })
      .Case<hlfir::DesignateOp>([&](hlfir::DesignateOp designateOp) {
        handleDesignateOp(tracking, opBuilder, funcOp, designateOp, sliceShiftMap);
      })
      .Case<hlfir::ApplyOp>([&](hlfir::ApplyOp applyOp){
        // example: %445 = "hlfir.apply"(%443, %arg9)  
        auto resultOperand = applyOp->getOpResult(0);
        auto refArr = applyOp.getOperand(0);
        tracking.valueMap.map(resultOperand, 
                              tracking.valueMap.lookup(refArr));
      })
      .Case<fir::LoadOp>([&](fir::LoadOp loadOp) {
        tracking.valueMap.map(loadOp->getResult(0), tracking.valueMap.lookup(loadOp->getOperand(0)));
      })
      .Case<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(
        [&](auto arithBinaryOp) {
          handleArithBinaryOp(tracking, opBuilder, funcOp, arithBinaryOp);
        }
      )
      .Case<math::SinOp, math::ExpOp>(
        [&](auto arithUnaryOp){
          handleArithUnaryOp(tracking, opBuilder, funcOp, arithUnaryOp);
        }
      )
      .Case<hlfir::MatmulOp, hlfir::DotProductOp, hlfir::TransposeOp>(
        [&](auto builtInOp) {
          handleBuiltinOperators(tracking, opBuilder, funcOp, builtInOp);
        }
      )
      .Case<hlfir::NoReassocOp>([&](hlfir::NoReassocOp nrop){
        assert(tracking.valueMap.contains(nrop.getOperand()) && "Operand of NoReassocOp is supposed to be in ValueMap!");
        // just ignore and pass to the result
        tracking.valueMap.map(nrop.getResult(), tracking.valueMap.lookup(nrop.getOperand()));
      })
      // TODO: including other cases!
      .Default([](auto) {});
}


static void terminateFunction(const TrackingInfo& tracking, OpBuilder& opBuilder, func::FuncOp& funcOp) {
  auto argNum = funcOp.getNumArguments();
  mlir::SmallVector<Value> returnValues;
  returnValues.reserve(argNum);
  for (unsigned int i = 0; i < funcOp.getNumArguments(); i ++) {
    auto currArg = funcOp.getArgument(i);
    auto trackedVal = tracking.valueMap.lookup(tracking.argsTrackingMap.lookup(currArg));
    returnValues.push_back(trackedVal);
  }

  // find the end of the last block
  auto loc = funcOp.getLoc();
  opBuilder.setInsertionPointToEnd(&funcOp.front());
  func::ReturnOp::create(opBuilder, loc, returnValues);
}

/// Parse the string into moduleOp and lowering, although the input is supposed to be a omp::targetOp,
/// but should also be compatible with following code.
func::FuncOp workdistributeToStableHLO(
  MLIRContext* context, const mlir::ModuleOp& moduleOp, 
  const llvm::DenseMap<Value, llvm::SmallVector<int>>& sliceShiftMap,
  llvm::DenseSet<Value>& privateValSet
) {
  PROFILE_SCOPE("workdistributeToStableHLO", Phase::LOWERING_TO_STABLEHLO);
  OpBuilder opBuilder(context);
  TrackingInfo trackingInfo;
  func::FuncOp stableHLOFuncOp;

  moduleOp->walk([&](func::FuncOp inputOp) {
    auto funcOp = createFunction(context, trackingInfo, inputOp);
    opBuilder.setInsertionPointToStart(&funcOp.front());
    inputOp->walk([&](Operation *op) {
      scanOperationsAndInserts(trackingInfo, opBuilder, funcOp, op, sliceShiftMap, privateValSet);
    });
    terminateFunction(trackingInfo, opBuilder, funcOp);
    stableHLOFuncOp = funcOp;
  });

  return stableHLOFuncOp;
}
