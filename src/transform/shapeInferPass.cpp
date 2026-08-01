#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include "../support/utilities.h"
#include "../support/profiler.h"
#include "passes.h"

using namespace mlir;

#define JIT_COMPUTE_ARG_ATTR_NAME "jit.compute_arg"
#define JIT_SLICE_SHIFT_ATTR_NAME "jit.slice_shift"
#define JIT_LITERAL_VAL_ATTR_NAME "jit.literal_val"

namespace {
struct ShapeInferPass
    : public PassWrapper<ShapeInferPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferPass)

  llvm::DenseMap<Value, int> valueMap;
  llvm::DenseMap<Value, llvm::SmallVector<int64_t>> sliceShiftMap;

  /// Fill some known values to the mlir and use existing passes to do constant
  /// propagation. Including:
  /// - CSE: Common Subexpression Elimination
  /// - Canonlicalize
  /// - SCCP: Sparse Conditional Constant Propagation
  /// Ref: https://mlir.llvm.org/docs/Passes/
  void preprocWithExistingPasses(OpBuilder opBuilder, func::FuncOp funcOp) {
    auto getSolidVal = [&](Value v) -> std::pair<int, bool> {
      auto it = valueMap.find(v);
      if (it != valueMap.end()) {
        return std::pair(it->getSecond(), true);
      }
      return std::pair(-1, false);
    };

    // Replace some known values with constant values, then lifiting the
    // propagation task to existing mlir passes.
    llvm::SmallVector<Operation *> opsToDelete;
    funcOp.walk([&](fir::LoadOp lop) {
      opBuilder.setInsertionPoint(lop);
      auto lopVal = getSolidVal(lop.getOperand());
      if (lopVal.second) {
        auto resValue = lop.getResult();
        auto resType = resValue.getType();
        if (!llvm::isa<mlir::IntegerType>(resType) &&
            !llvm::isa<mlir::IndexType>(resType)) {
          return;
        }

        arith::ConstantIntOp cop = arith::ConstantIntOp::create(
            opBuilder, funcOp.getLoc(), resValue.getType(), lopVal.first);
        lop.replaceAllUsesWith(cop.getResult());
        assert(lop.use_empty() && "Still been used!");
        opsToDelete.push_back(lop);
      }
    });

    for (auto *op : opsToDelete) {
      op->erase();
    }

    // llvm::dbgs() << "\n ## after replace known values\n";

    // Run passes
    mlir::PassManager pm(funcOp.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCCPPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(funcOp))) {
      llvm::errs() << "[Fail] Fail to run passes on funcOp!\n";
      std::exit(EXIT_FAILURE);
    }
    return;
  }

  // Not a roboust transformation but works for now.
  void shapeInferenceInternal(OpBuilder opBuilder, func::FuncOp funcOp) {
    // mapping from value to shape (a vector of each dimension)
    llvm::DenseMap<Value, llvm::SmallVector<int64_t>> shapeMap;
    // tracking shape constant
    llvm::DenseMap<Value, int64_t> constTrackingMap;

    funcOp.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<arith::ConstantOp>([&](arith::ConstantOp cop) {
            auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cop.getValue());
            if (intAttr) {
              constTrackingMap.insert(
                  std::pair<Value, int64_t>(cop.getResult(), intAttr.getInt()));
            }
          })
          .Case<fir::ShapeOp>([&](fir::ShapeOp sop) {
            // %0 = fir.shape %c1000, %c1000 : (index, index) -> !fir.shape<2>
            llvm::SmallVector<int64_t> sizes; // {1000, 1000} in this example
            sizes.reserve(sop.getNumOperands());
            for (int i = 0; i < sop.getNumOperands(); i++) {
              auto opr = sop.getOperand(i);
              assert(constTrackingMap.contains(opr) &&
                     "Operand static value should be known!");
              sizes.push_back(constTrackingMap.at(opr));
            }
            shapeMap.insert(
                std::pair(sop.getResult(), sizes)); // %0 -> {1000, 1000}
          })
          .Case<fir::ShapeShiftOp>([&](fir::ShapeShiftOp ssOp) {
            // In Fortran, the idx can start from any number, often we have
            // shapeShiftOp %10 = fir.shape_shift %c-1, %c964, %c-1, %c965 :
            // (index, index, index, index) -> !fir.shapeshift<2> The shape is
            // rank=2:
            // - first rank: lower bound: %c-1, len: %c964
            // - second rank: lower bound: %c-1, len: %c965
            //
            // TODO: the lower bound should also be stored for later usage: for
            // example, when translating from `hlfir::desinateOp` to
            // `stablehlo::slicingOp`
            llvm::SmallVector<int64_t> shapeSizes;
            llvm::SmallVector<int64_t> shapeShifts;
            assert(ssOp.getNumOperands() % 2 == 0 &&
                   "ShapeShift should have even number of operands!");
            shapeSizes.reserve(ssOp.getNumOperands() / 2);

            for (int i = 0; i < ssOp.getNumOperands(); i++) {
              auto opr = ssOp.getOperand(i);
              assert(constTrackingMap.contains(opr) &&
                     "Operand static value of shapeShift should be known!");
              if (i % 2 == 0) {
                auto idxLowerBound = constTrackingMap.at(opr);
                shapeShifts.push_back(idxLowerBound);
              } else {
                auto dimSize = constTrackingMap.at(opr);
                shapeSizes.push_back(dimSize);
              }
            }
            shapeMap.insert(
                std::pair(ssOp.getResult(), shapeSizes)); // %10 -> {%c964, %c965}
            sliceShiftMap.insert(
                std::pair(ssOp.getResult(), shapeShifts)); // %10 -> {%c-1, %c-1}
          })
          .Case<hlfir::DeclareOp>([&](hlfir::DeclareOp dop) {
            // Example: %1:2 = hlfir.declare %arg0(%0) {uniq_name =
            // "_QFFcoexecute_aEz"} : (!fir.ref<!fir.array<?x?xf64>>,
            // !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>,
            // !fir.ref<!fir.array<?x?xf64>>) 
            //  
            // Expect: 
            // %1:2 = hlfir.declare
            // %arg0(%0) {uniq_name = "_QFFcoexecute_aEz"} :
            // (!fir.ref<!fir.array<1000x1000xf64>>, !fir.shape<2>) ->
            // (!fir.box<!fir.array<1000x1000xf64>>,
            // !fir.ref<!fir.array<1000x1000xf64>>)
            if (isDynamicShape(dop.getResult(0).getType())) {
              assert(shapeMap.contains(dop.getShape()) &&
                     "The shape of the declareOp has not been added!!!!");
              auto staticShape = shapeMap.at(dop.getShape());

              // propagate the shape of the results
              for (int i = 0; i < dop.getNumResults(); i++) {
                shapeMap.insert(std::pair(dop.getResult(i), staticShape));
              }
              shapeMap.insert(std::pair(dop.getMemref(), staticShape));

              // Insert the new declareOp
              opBuilder.setInsertionPoint(dop);
              auto ndop = hlfir::DeclareOp::create(
                  opBuilder, dop.getLoc(),
                  convertToStaticShape(dop.getResult(0).getType(), staticShape),
                  convertToStaticShape(dop.getResult(1).getType(), staticShape),
                  dop.getMemref(), dop.getShape(), dop.getTypeparams(),
                  dop.getDummyScope(), dop.getStorage(),
                  dop.getStorageOffsetAttr(), dop.getUniqNameAttr(),
                  dop.getFortranAttrsAttr(), dop.getDataAttrAttr(), 
                  dop.getSkipReboxAttr()
              );
              dop.replaceAllUsesWith(ndop.getResults());
              dop.erase();
            }
          })
          .Case<fir::AllocaOp>([&](fir::AllocaOp allocaOp) {
            if (isDynamicShape(allocaOp.getResult().getType())) {
              llvm::SmallVector<int64_t> staticShape;
              auto shapeOperands = allocaOp.getShape();
              staticShape.reserve(shapeOperands.size());

              for (auto opr : shapeOperands) {
                assert(constTrackingMap.contains(opr) &&
                       "Operand static value for allocaOp should be known!");
                staticShape.push_back(constTrackingMap.at(opr));
              }

              shapeMap.insert(std::pair(allocaOp.getResult(), staticShape));

              Type newInType =
                  convertToStaticShape(allocaOp.getInType(), staticShape);

              opBuilder.setInsertionPoint(allocaOp);
              auto staticAllocaOp = fir::AllocaOp::create(
                  opBuilder, allocaOp.getLoc(), newInType,
                  allocaOp.getUniqName().value_or(""),
                  allocaOp.getBindcName().value_or(""), allocaOp.getPinned(),
                  allocaOp.getTypeparams(), ValueRange{});

              allocaOp.replaceAllUsesWith(staticAllocaOp.getResult());
              allocaOp.erase();
            }
          })
          .Case<hlfir::DesignateOp>([&](hlfir::DesignateOp dop) {
            // %5 = hlfir.designate %3#0 (%c1:%c5:%c1)  shape %4 :
            // (!fir.box<!fir.array<?xf64>>, index, index, index, !fir.shape<1>)
            // -> !fir.box<!fir.array<?xf64>> In the above example, it creates a
            // part-ref of %3#0, with starting index %c1, end index %c5 and stride
            // %c1 in such case, the shape will be different But it can also be
            // something a simple form, referring just a single value of it:
            // example: %451 = "hlfir.designate"(%447#0, %arg9)
            if (isDynamicShape(dop.getResult().getType())) {
              auto shapeVal = dop.getShape();
              auto it = shapeMap.find(shapeVal);
              assert(it != shapeMap.end());
              auto shapeInfo = it->getSecond();
              opBuilder.setInsertionPoint(dop);
              auto neoDop = hlfir::DesignateOp::create(
                  opBuilder, dop.getLoc(),
                  convertToStaticShape(dop.getResult().getType(), shapeInfo),
                  dop.getMemref(), dop.getComponentAttr(),
                  dop.getComponentShape(), dop.getIndices(),
                  dop.getIsTripletAttr(), dop.getSubstring(),
                  dop.getComplexPartAttr(), dop.getShape(), dop.getTypeparams(),
                  dop.getFortranAttrsAttr());
              dop.replaceAllUsesWith(neoDop.getResult());
              dop.erase();
            }
          })
          .Case<hlfir::ElementalOp>([&](hlfir::ElementalOp eop) {
            // Example: %6 = hlfir.elemental %0 unordered : (!fir.shape<2>) ->
            // !hlfir.expr<?x?xf64> { static ElementalOp create(::mlir::OpBuilder
            // &builder, ::mlir::Location location, mlir::Type result_type,
            // mlir::Value shape, mlir::Value mold = {}, mlir::ValueRange
            // typeparams = {}, bool isUnordered = false);
            if (isDynamicShape(eop.getResult().getType()) &&
                !isDynamicShape(eop.getShape().getType())) {
              assert(shapeMap.contains(eop.getShape()) &&
                     "The shape of the elementalOp should be known");
              auto oldResType = eop.getResult().getType();
              auto newResType =
                  convertToStaticShape(oldResType, shapeMap.at(eop.getShape()));
              opBuilder.setInsertionPoint(eop);
              auto neop = hlfir::ElementalOp::create(
                  opBuilder, funcOp.getLoc(), newResType, eop.getShape(),
                  eop.getMold(), eop.getTypeparams(), eop.isOrdered());
              shapeMap.insert(
                  std::pair(neop.getResult(), shapeMap.at(eop.getShape())));
              neop.getRegion().takeBody(eop.getRegion());
              eop.replaceAllUsesWith(neop.getResult());
              eop.erase();
            }
          })
          .Case<hlfir::SumOp>([&](hlfir::SumOp sumOp) {
            // %17 = "hlfir.sum"(%16, %2) <{fastmath = #arith.fastmath<contract>,
            // operandSegmentSizes = array<i32: 1, 1, 0>}> :
            // (!hlfir.expr<100x128xf64>, i32) -> !hlfir.expr<?xf64>
            if (isDynamicShape(sumOp.getResult().getType())) {
              auto arrayVal = sumOp.getArray();
              assert(shapeMap.contains(arrayVal));
              if (!shapeMap.contains(arrayVal)) {
                return;
              }
              auto inputShape = shapeMap.at(arrayVal);
              llvm::SmallVector<int64_t> outputShape;

              if (sumOp.getDim()) {
                auto dimVal = sumOp.getDim();
                int64_t dimIdx = -1;

                if (constTrackingMap.contains(dimVal)) {
                  dimIdx = constTrackingMap.at(dimVal);
                } else if (auto constOp =
                               llvm::dyn_cast_or_null<arith::ConstantOp>(
                                   dimVal.getDefiningOp())) {
                  if (auto intAttr =
                          llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
                    dimIdx = intAttr.getInt();
                  }
                }

                if (dimIdx != -1) {
                  int64_t zeroBasedDim = dimIdx - 1;
                  for (size_t i = 0; i < inputShape.size(); ++i) {
                    if (i != zeroBasedDim) {
                      outputShape.push_back(inputShape[i]);
                    }
                  }
                } else {
                  return;
                }
              } else {
                // DO NOTHING
              }
              shapeMap.insert(std::pair(sumOp.getResult(), outputShape));
              Type newResType =
                  convertToStaticShape(sumOp.getResult().getType(), outputShape);
              opBuilder.setInsertionPoint(sumOp);
              auto newSumOp = hlfir::SumOp::create(
                  opBuilder, funcOp->getLoc(), newResType, sumOp.getArray(),
                  sumOp.getDim(), sumOp.getMask(), sumOp.getFastmathAttr());
              sumOp.replaceAllUsesWith(newSumOp.getResult());
              sumOp.erase();
            }
          })
          .Case<func::FuncOp>([&](func::FuncOp fop) {
            auto funcType = funcOp.getFunctionType();
            auto inputTypes = llvm::to_vector(funcType.getInputs());
            auto oldRes = funcType.getResults();

            Block &entryBlock = fop.front();
            for (int i = 0; i < entryBlock.getNumArguments(); i++) {
              auto arg = entryBlock.getArgument(i);
              if (isDynamicShape(arg.getType())) {
                assert(shapeMap.contains(arg) && "Arg Shape should be known!");
                auto staticShape = shapeMap.at(arg);
                arg.setType(convertToStaticShape(arg.getType(), staticShape));
                inputTypes[i] = convertToStaticShape(inputTypes[i], staticShape);
              }
            }
            auto newFuncType =
                FunctionType::get(funcOp.getContext(), inputTypes, oldRes);
            funcOp.setType(newFuncType);
          })
          .Default([](auto) {});
    });
    return;
  }

  void setSliceShiftAsAttr(OpBuilder opBuilder, func::FuncOp funcOp) {
    funcOp -> walk([&](fir::ShapeShiftOp ssOp){
      auto it = sliceShiftMap.find(ssOp.getResult());
      assert(it!= sliceShiftMap.end() 
             && "All slice shift Op Value is supposed to stored in sliceShiftMap!\n");
      ArrayRef<int64_t> shifts(it->getSecond());
      ssOp->setAttr(JIT_SLICE_SHIFT_ATTR_NAME, opBuilder.getDenseI64ArrayAttr(shifts));
      return;
    });  
  }

  StringRef getArgument() const override { 
    return "jforce-shape-infer"; 
  }

  void runOnOperation() override {
    PROFILE_SCOPE("shape infer", Phase::LOWERING_SHAPE_INFER);
    func::FuncOp funcOp = getOperation();
    MLIRContext* ctx = funcOp.getContext();
    OpBuilder opBuilder(ctx);

    // some parameters containing the shape info are passed as pointer like
    for (int i = 0; i < funcOp.getNumArguments(); i++) {
      auto isComputeArg = funcOp.getArgAttr(i, JIT_COMPUTE_ARG_ATTR_NAME);
      if (!isComputeArg) {
        auto intAttr = funcOp.getArgAttrOfType<mlir::IntegerAttr>(i, JIT_LITERAL_VAL_ATTR_NAME);
        if (intAttr) {
          // `intAttr` is the literal address, need to recover to specific number.
          auto argTy = funcOp.getArgumentTypes()[i];
          auto eleTy = getDTypeFromValueType(argTy);
          assert(eleTy == DType::I32 && "Supposed to be shape size!\n");
          auto eleVal = extractLiteralPtr(intAttr.getInt(), eleTy);
          assert(eleVal.returnedType == DType::I32);
          valueMap.insert(std::pair<Value, int>(funcOp.getArgument(i), eleVal.valI32));
        }
      }
    };

    preprocWithExistingPasses(opBuilder, funcOp);
    shapeInferenceInternal(opBuilder, funcOp);
    setSliceShiftAsAttr(opBuilder, funcOp); 
  }
};
} // namespace

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createShapeInferPass() {
    return std::make_unique<ShapeInferPass>();
  }

  void registerShapeInferPass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createShapeInferPass();});
  };
}
