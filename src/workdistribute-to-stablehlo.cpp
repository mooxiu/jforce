#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstdlib>
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
#include <omp.h>

using namespace mlir;

/**
 * valueMap, argsTrackingMaps are 2 maps we'll keep updating when scanning 
 * - valueMap: tracking the each operand of FIR pointing to the value of each operand in Stablehlo function
 * - argsTrackingMap: tracking the current value of arguments of stablehlo pointing to, practically a reverse map of `valueMap`
*/
struct TrackingInfo {
public:
  mlir::IRMapping valueMap;
  mlir::IRMapping argsTrackingMap;
};

/**
  Example of source type:
  "!fir.ref<!fir.array<10xf32>>": convert to "tensor<10xf32>"
  "!fir.ref<f32>": convert to "tensor<f32>"
 */
static RankedTensorType convertBufferTyToTensorTy(mlir::Type srcTy) {
  // If it's already a tensor type, then no need to convert
  if (llvm::isa<RankedTensorType>(srcTy)) {
    return llvm::dyn_cast<RankedTensorType>(srcTy);
  }

  if (auto refType = mlir::dyn_cast<fir::ReferenceType>(srcTy)) {
    srcTy = refType.getEleTy();
  }
  auto seqTy = mlir::dyn_cast<fir::SequenceType>(srcTy);
  if (!seqTy) {
    // this is a scalar
    auto scalarTensorType = RankedTensorType::get({}, srcTy);
    return scalarTensorType;
  }
  auto arrTensorType = RankedTensorType::get(seqTy.getShape(), seqTy.getEleTy());
  return arrTensorType;
}

/**
  We're dealing with TargetOp like following:
  > omp.target map_entries(%141 -> %arg0, %142 -> %arg1, %145 -> %arg2 :
  !fir.ref<!fir.array<10xf32>>, !fir.ref<f32>, !fir.ref<!fir.array<10xf32>>) {

  In which, "%141, %142, %145" is out values, they will be used when get the
  value and call stablehlo function;
  "%arg0, %arg1, %arg2" are the values we need to track.
 */
static func::FuncOp createFunction(mlir::MLIRContext &context,
                                   TrackingInfo &tracking,
                                   const func::FuncOp& inputOp) {
  auto &firstRegion = inputOp->getRegion(0);
  auto &block = firstRegion.getBlocks().front();

  mlir::SmallVector<Type> inputTypes, outputTypes;
  for (auto arg : block.getArguments()) {
    inputTypes.push_back(convertBufferTyToTensorTy(arg.getType()));
    outputTypes.push_back(convertBufferTyToTensorTy(arg.getType()));
  }

  auto funcType = mlir::FunctionType::get(&context, inputTypes, outputTypes);
  auto funcOp =
      func::FuncOp::create(inputOp->getLoc(), "main", funcType, {});
  // we need to update the valueMap!
  funcOp.addEntryBlock();

  for (unsigned int i = 0; i < block.getNumArguments(); i++) {
    Value oldArgOperand = block.getArgument(i);
    Value newArgOperand = funcOp.getArgument(i);
    tracking.valueMap.map(oldArgOperand, newArgOperand); 
    tracking.argsTrackingMap.map(newArgOperand, oldArgOperand);
  }
  return funcOp;
}


/**
* Only support increase one dimension right now, for example:
* - tensor<f32> -> tensor<10xf32>
* - tensor<10xf32> -> tensor<10x10xf32>
* 
* Ref: https://openxla.org/stablehlo/spec#broadcast_in_dim
*/
static void handleArithBinaryOp(TrackingInfo& tracking, 
                                OpBuilder &opBuilder, 
                                func::FuncOp& funcOp, 
                                Operation* arithOp) {
  assert(arithOp->hasTrait<mlir::OpTrait::OneResult>());
  assert(arithOp->hasTrait<mlir::OpTrait::NOperands<2>::Impl>());
  Value operand1 = arithOp->getOperand(0);
  Value operand2 = arithOp->getOperand(1);
  Value result = arithOp->getResult(0);

  Value operand1Src = tracking.valueMap.lookup(operand1);
  Value operand2Src = tracking.valueMap.lookup(operand2);


  RankedTensorType o1Type = convertBufferTyToTensorTy(operand1Src.getType()); 
  RankedTensorType o2Type = convertBufferTyToTensorTy(operand2Src.getType()); 
  assert(o1Type.hasRank() && o2Type.hasRank());

  Value largerOperand, smallerOperand; 
  if (o1Type.getRank() >= o2Type.getRank()) {
    largerOperand = operand1Src;
    smallerOperand = operand2Src;
  } else {
    largerOperand = operand2Src;
    smallerOperand = operand1Src;
  }
  RankedTensorType targetType = convertBufferTyToTensorTy(largerOperand.getType());

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
  .Default([](auto){
      llvm::errs() << "Unknown arith operation! \n";
      return;
    });

  tracking.valueMap.map(result, stablehloRes);
  tracking.argsTrackingMap.map(stablehloRes, result);
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
      stablehlo::DotDimensionNumbersAttr attr = {};
      ArrayAttr precisionConfig = {};
      stablehlo::DotAlgorithmAttr algoAttr = {};
      auto stablehloDotProductOp = stablehlo::DotGeneralOp::create(opBuilder, funcOp->getLoc(), scalarType, lhs, rhs, attr, precisionConfig, algoAttr);
      tracking.valueMap.map(dpOp.getResult(), stablehloDotProductOp.getResult());
    })
    .Case<hlfir::MatmulOp>([&](hlfir::MatmulOp mmOp){
      // %36 = hlfir.matmul %33#0 %35#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>) -> !hlfir.expr<?x?xf64>
      auto stablehloMulOp = stablehlo::MulOp::create(
        opBuilder, 
        funcOp.getLoc(), 
        mmOp.getResult().getType(), 
        tracking.valueMap.lookup(mmOp.getOperand(0)), 
        tracking.valueMap.lookup(mmOp.getOperand(1))
      );
      tracking.valueMap.map(mmOp.getResult(), stablehloMulOp.getResult());
    })
    .Case<hlfir::TransposeOp>([&](hlfir::TransposeOp tOp){
      // %24 = hlfir.transpose %23#0 : (!fir.box<!fir.array<?x?xf64>>) -> !hlfir.expr<?x?xf64>
      auto stablehloTransposeOp = stablehlo::TransposeOp::create(
        opBuilder, 
        funcOp.getLoc(),
        tOp.getResult().getType(),
        tracking.valueMap.lookup(tOp.getOperand())
      );
      tracking.valueMap.map(tOp.getResult(), stablehloTransposeOp.getResult());
    })
    .Default([](auto){
      llvm::errs() << "Not Supported Builtin Operators!\n";
      std::exit(EXIT_FAILURE);
    });
}

static void scanOperationsAndInserts(TrackingInfo& tracking,
                                     OpBuilder &opBuilder, 
                                     func::FuncOp& funcOp,
                                     Operation *op) {
  llvm::TypeSwitch<Operation *>(op)
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
        // example: hlfir.assign %155 to %150#0 
        // Assign A to B
        assert(assignOp.getNumOperands() == 2 && "Fail to assert assignOp has 2 Operands!");
        auto assignFromOperand = assignOp.getOperand(0); 
        auto assignToOperand = assignOp.getOperand(1); 

        // value: B should tracking the same value as A
        tracking.valueMap.map(assignToOperand, tracking.valueMap.lookup(assignFromOperand));  

        // argsTrackingMap: the arg which is tracking B now should tracking A?
        tracking.argsTrackingMap.map(tracking.valueMap.lookup(assignToOperand), assignFromOperand);
      })
      .Case<hlfir::DeclareOp>([&](hlfir::DeclareOp declareOp) {
        auto declaredOprand = declareOp.getOperand(0);
        for (unsigned int i = 0; i < declareOp->getNumResults(); i++) {
          tracking.valueMap.map(declareOp->getOpResult(i), tracking.valueMap.lookup(declaredOprand));
        }
        tracking.argsTrackingMap.map(tracking.valueMap.lookup(declaredOprand), declareOp->getOpResult(0));
      })
      .Case<hlfir::DesignateOp>([&](hlfir::DesignateOp designateOp) {
        // example: %451 = "hlfir.designate"(%447#0, %arg9) 
        auto resultOperand = designateOp->getOpResult(0);

        auto refArr = designateOp.getMemref();

        tracking.valueMap.map(resultOperand, 
                              tracking.valueMap.lookup(refArr));
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
      .Case<arith::AddFOp>([&](arith::AddFOp addFOp) {
        handleArithBinaryOp(tracking, opBuilder, funcOp, addFOp);
      })
      .Case<arith::MulFOp>([&](arith::MulFOp mulFOp){
        handleArithBinaryOp(tracking, opBuilder, funcOp, mulFOp);
      })
      .Case<hlfir::MatmulOp>([&](hlfir::MatmulOp matmulOp) {
        handleBuiltinOperators(tracking, opBuilder, funcOp, matmulOp);
      })
      .Case<hlfir::DotProductOp>([&](hlfir::DotProductOp dotProductOp){
        handleBuiltinOperators(tracking, opBuilder, funcOp, dotProductOp);
      })
      .Case<hlfir::TransposeOp>([&](hlfir::TransposeOp transposeOp){
        handleBuiltinOperators(tracking, opBuilder, funcOp, transposeOp);
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

// Parse the string into moduleOp and lowering, although the input is supposed to be a omp::targetOp,
// but should also be compatible with following code.
func::FuncOp workdistributeToStableHLO(MLIRContext& context, const mlir::ModuleOp& moduleOp) {
  OpBuilder opBuilder(&context);
  TrackingInfo trackingInfo;
  func::FuncOp stableHLOFuncOp;

  moduleOp->walk([&](func::FuncOp inputOp) {
    auto funcOp = createFunction(context, trackingInfo, inputOp);
    opBuilder.setInsertionPointToStart(&funcOp.front());
    inputOp->walk([&](Operation *op) {
      scanOperationsAndInserts(trackingInfo, opBuilder, funcOp, op);
    });
    terminateFunction(trackingInfo, opBuilder, funcOp);
    stableHLOFuncOp = funcOp;
  });

  return stableHLOFuncOp;
}
