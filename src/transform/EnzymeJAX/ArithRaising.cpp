/// Copied and Adapted from Enzyme-JAX


//===- ArithRaising.cpp - Raise to Arith dialect --------------------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to raise operations to arith dialect.
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "../Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace {
struct ArithRaisingPass
    : public mlir::PassWrapper<ArithRaisingPass, mlir::OperationPass<mlir::func::FuncOp>> {

  void getDependentDialects(mlir::DialectRegistry & registry) const override {
    return registry.insert<stablehlo::StablehloDialect>(); 
  }

  StringRef getArgument() const override {
    return "enzyme-arith-raise";
  }


  void runOnOperation() override {
  auto op = getOperation();

#define RAISE_BINARY(BinaryOp, StableHLOOp)                                    \
  op->walk([=](BinaryOp origOp) {                                              \
    if (!isa<RankedTensorType>(origOp->getResult(0).getType()))                \
      return;                                                                  \
    OpBuilder builder(origOp);                                                 \
    Value newOp;                                                               \
    newOp =                                                                    \
          StableHLOOp::create(builder, origOp.getLoc(), origOp->getOperand(0), \
                              origOp->getOperand(1))                           \
              .getResult();                                                    \
    origOp.replaceAllUsesWith(newOp);                                          \
    origOp.erase();                                                            \
  });

    RAISE_BINARY(arith::AddFOp, stablehlo::AddOp);
    RAISE_BINARY(arith::AddIOp, stablehlo::AddOp);
    RAISE_BINARY(arith::SubFOp, stablehlo::SubtractOp);
    RAISE_BINARY(arith::SubIOp, stablehlo::SubtractOp);
    RAISE_BINARY(arith::MulFOp, stablehlo::MulOp);
    RAISE_BINARY(arith::MulIOp, stablehlo::MulOp);
    RAISE_BINARY(arith::DivFOp, stablehlo::DivOp);
    RAISE_BINARY(arith::DivSIOp, stablehlo::DivOp);
    RAISE_BINARY(arith::DivUIOp, stablehlo::DivOp);
    RAISE_BINARY(arith::MaximumFOp, stablehlo::MaxOp);
    RAISE_BINARY(arith::MaxSIOp, stablehlo::MaxOp);
    RAISE_BINARY(arith::MaxUIOp, stablehlo::MaxOp);
    RAISE_BINARY(arith::MinimumFOp, stablehlo::MinOp);
    RAISE_BINARY(arith::MinSIOp, stablehlo::MinOp);
    RAISE_BINARY(arith::MinUIOp, stablehlo::MinOp);
    RAISE_BINARY(arith::ShLIOp, stablehlo::ShiftLeftOp);
    RAISE_BINARY(arith::ShRSIOp, stablehlo::ShiftRightArithmeticOp);
    RAISE_BINARY(arith::ShRUIOp, stablehlo::ShiftRightLogicalOp);
    RAISE_BINARY(complex::AddOp, stablehlo::AddOp);
    RAISE_BINARY(arith::AndIOp, stablehlo::AndOp);
    RAISE_BINARY(arith::OrIOp, stablehlo::OrOp);
    RAISE_BINARY(arith::XOrIOp, stablehlo::XorOp);
    RAISE_BINARY(math::PowFOp, stablehlo::PowOp);
    RAISE_BINARY(arith::RemFOp, stablehlo::RemOp);
    RAISE_BINARY(arith::RemUIOp, stablehlo::RemOp);

#undef RAISE_BINARY

#define RAISE_UNARY(InputOp, StableHLOOp)                                      \
  op->walk([=](InputOp inpOp) {                                                \
    if (!isa<RankedTensorType>(inpOp.getType()))                               \
      return;                                                                  \
    OpBuilder builder(inpOp);                                                  \
    Value newAddOp;                                                            \
    newAddOp =                                                                 \
          StableHLOOp::create(builder, inpOp.getLoc(), inpOp->getOperand(0));  \
    inpOp.replaceAllUsesWith(newAddOp);                                        \
    inpOp.erase();                                                             \
  });

    RAISE_UNARY(math::SinOp, stablehlo::SineOp);
    RAISE_UNARY(math::CosOp, stablehlo::CosineOp);
    RAISE_UNARY(math::LogOp, stablehlo::LogOp);
    RAISE_UNARY(math::Log1pOp, stablehlo::Log1pOp);
    RAISE_UNARY(math::ExpOp, stablehlo::ExpOp);
    RAISE_UNARY(math::ExpM1Op, stablehlo::Expm1Op);
    RAISE_UNARY(math::TanhOp, stablehlo::TanhOp);
    RAISE_UNARY(math::SqrtOp, stablehlo::SqrtOp);
    RAISE_UNARY(math::RsqrtOp, stablehlo::RsqrtOp);
    RAISE_UNARY(math::CbrtOp, stablehlo::CbrtOp);
    RAISE_UNARY(math::AbsFOp, stablehlo::AbsOp);
    RAISE_UNARY(math::IsFiniteOp, stablehlo::IsFiniteOp);
    RAISE_UNARY(math::CeilOp, stablehlo::CeilOp);
    RAISE_UNARY(math::FloorOp, stablehlo::FloorOp);
    RAISE_UNARY(arith::NegFOp, stablehlo::NegOp);

#undef RAISE_UNARY

    // op->walk([=](arith::BitcastOp op) {
    //   auto ty = dyn_cast<RankedTensorType>(op.getResult().getType());
    //   if (!ty)
    //     return;
    //
    //   size_t outSize =
    //       cast<AutoDiffTypeInterface>(ty.getElementType()).getApproxSize();
    //   size_t inSize = cast<AutoDiffTypeInterface>(
    //                       cast<RankedTensorType>(op.getOperand().getType())
    //                           .getElementType())
    //                       .getApproxSize();
    //
    //   OpBuilder builder(op);
    //   Value res;
    //   if (outSize == inSize) {
    //     res = stablehlo::BitcastConvertOp::create(builder, op.getLoc(), ty,
    //                                               op.getIn());
    //   } else if (outSize < inSize) {
    //     SmallVector<int64_t> dims2 = llvm::to_vector(ty.getShape());
    //     auto oidx = dims2.size();
    //     dims2.push_back(inSize / outSize);
    //     if (oidx != 0 && dims2[oidx - 1] != ShapedType::kDynamic) {
    //       dims2[oidx - 1] /= inSize / outSize;
    //     }
    //     res = stablehlo::BitcastConvertOp::create(
    //         builder, op.getLoc(),
    //         RankedTensorType::get(dims2, ty.getElementType()), op.getIn());
    //     bool anyDynamic = false;
    //     for (auto idx : dims2) {
    //       if (idx == ShapedType::kDynamic) {
    //         anyDynamic = true;
    //         break;
    //       }
    //     }
    //     if (anyDynamic) {
    //       SmallVector<Value> vals;
    //       for (size_t i = 0; i < ty.getShape().size(); i++) {
    //         auto val = stablehlo::GetDimensionSizeOp::create(
    //             builder, op.getLoc(), op.getIn(), i);
    //         Value vval = val;
    //         if (i == ty.getShape().size() - 1) {
    //           auto cst = arith::ConstantOp::create(
    //               builder, op.getLoc(), val.getType(),
    //               cast<ElementsAttr>(
    //                   makeAttr(val.getType(), inSize / outSize)));
    //           vval = stablehlo::MulOp::create(builder, op.getLoc(), vval, cst);
    //         }
    //         vval = stablehlo::ReshapeOp::create(
    //             builder, op.getLoc(),
    //             RankedTensorType::get({1}, val.getType().getElementType()),
    //             vval);
    //         vals.push_back(vval);
    //       }
    //
    //       auto idxs =
    //           stablehlo::ConcatenateOp::create(builder, op.getLoc(), vals, 0);
    //       res = stablehlo::DynamicReshapeOp::create(builder, op.getLoc(), ty,
    //                                                 res, idxs);
    //     } else {
    //       res = stablehlo::ReshapeOp::create(builder, op.getLoc(), ty, res);
    //     }
    //   } else {
    //     SmallVector<int64_t> dims2 = llvm::to_vector(ty.getShape());
    //     auto oidx = dims2.size();
    //     dims2.push_back(outSize / inSize);
    //     if (oidx != 0 && dims2[oidx - 1] != ShapedType::kDynamic) {
    //       dims2[oidx - 1] /= outSize / inSize;
    //     }
    //     res = stablehlo::ReshapeOp::create(
    //         builder, op.getLoc(),
    //         RankedTensorType::get(
    //             dims2, cast<RankedTensorType>(op.getOperand().getType())
    //                        .getElementType()),
    //         op.getIn());
    //     res =
    //         stablehlo::BitcastConvertOp::create(builder, op.getLoc(), ty, res);
    //   }
    //   op.replaceAllUsesWith(res);
    //   op.erase();
    // });

    // op->walk([=](arith::ConvertFOp cvtOp) {
    //   auto ty = dyn_cast<RankedTensorType>(cvtOp.getResult().getType());
    //   if (!true || !ty)
    //     return;
    //
    //   OpBuilder builder(cvtOp);
    //   auto res = stablehlo::ConvertOp::create(builder, cvtOp.getLoc(), ty,
    //                                           cvtOp.getIn());
    //   cvtOp.replaceAllUsesWith(res.getResult());
    //   cvtOp.erase();
    // });
    op->walk([=](arith::TruncFOp truncOp) {
      auto ty = dyn_cast<RankedTensorType>(truncOp.getResult().getType());
      if (!true || !ty)
        return;

      OpBuilder builder(truncOp);
      auto res = stablehlo::ConvertOp::create(builder, truncOp.getLoc(), ty,
                                              truncOp.getIn());
      truncOp.replaceAllUsesWith(res.getResult());
      truncOp.erase();
    });
    op->walk([=](arith::ExtFOp truncOp) {
      auto ty = dyn_cast<RankedTensorType>(truncOp.getResult().getType());
      if (!true || !ty)
        return;

      OpBuilder builder(truncOp);
      auto res = stablehlo::ConvertOp::create(builder, truncOp.getLoc(), ty,
                                              truncOp.getIn());
      truncOp.replaceAllUsesWith(res.getResult());
      truncOp.erase();
    });
    // TODO: either SI or UI is wrong
    op->walk([=](arith::ExtUIOp cvtOp) {
      auto ty = dyn_cast<RankedTensorType>(cvtOp.getResult().getType());
      if (!true || !ty)
        return;

      OpBuilder builder(cvtOp);
      auto res = stablehlo::ConvertOp::create(builder, cvtOp.getLoc(), ty,
                                              cvtOp.getIn());
      cvtOp.replaceAllUsesWith(res.getResult());
      cvtOp.erase();
    });
    op->walk([=](arith::ExtSIOp cvtOp) {
      auto ty = dyn_cast<RankedTensorType>(cvtOp.getResult().getType());
      if (!true || !ty)
        return;

      OpBuilder builder(cvtOp);
      auto res = stablehlo::ConvertOp::create(builder, cvtOp.getLoc(), ty,
                                              cvtOp.getIn());
      cvtOp.replaceAllUsesWith(res.getResult());
      cvtOp.erase();
    });
    op->walk([=](arith::TruncIOp truncOp) {
      auto ty = dyn_cast<RankedTensorType>(truncOp.getResult().getType());
      if (!true || !ty)
        return;

      OpBuilder builder(truncOp);
      auto res = stablehlo::ConvertOp::create(builder, truncOp.getLoc(), ty,
                                              truncOp.getIn());
      truncOp.replaceAllUsesWith(res.getResult());
      truncOp.erase();
    });
    op->walk([=](math::FmaOp fma) {
      auto ty = dyn_cast<RankedTensorType>(fma.getResult().getType());
      if (!true || !ty)
        return;

      OpBuilder builder(fma);
      auto res = stablehlo::MulOp::create(builder, fma.getLoc(),
                                          fma.getOperand(0), fma.getOperand(1));
      auto res2 = stablehlo::AddOp::create(builder, fma.getLoc(), res,
                                           fma.getOperand(2));
      fma.replaceAllUsesWith(res2.getResult());
      fma.erase();
    });

    op->walk([=](math::CopySignOp copySignOp) {
      auto ty = dyn_cast<RankedTensorType>(copySignOp.getResult().getType());
      if (!true || !ty)
        return;

      // The copysign returns a value with the magnitude of the first operand
      // and the sign of the second operand.
      OpBuilder builder(copySignOp);
      auto loc = copySignOp.getLoc();
      Value val = copySignOp.getLhs();
      Value sign = copySignOp.getRhs();
      Attribute constAttr = FloatAttr::get(ty.getElementType(), 0);
      Value zero = stablehlo::ConstantOp::create(
          builder, loc, ty, SplatElementsAttr::get(ty, constAttr));
      Value signPositive = stablehlo::CompareOp::create(
          builder, loc, sign, zero, stablehlo::ComparisonDirection::GE);
      Value valPositive = stablehlo::CompareOp::create(
          builder, loc, val, zero, stablehlo::ComparisonDirection::GE);
      Value notSameSign =
          stablehlo::XorOp::create(builder, loc, signPositive, valPositive);
      Value negVal = stablehlo::NegOp::create(builder, loc, val);
      Value res =
          stablehlo::SelectOp::create(builder, loc, notSameSign, negVal, val);

      copySignOp.replaceAllUsesWith(res);
      copySignOp.erase();
    });

    op->walk([=](math::AtanOp atanOp) {
      // atan %a -> atan2(%a, 1.0)
      auto ty = dyn_cast<RankedTensorType>(atanOp.getResult().getType());
      if (!true || !ty)
        return;

      OpBuilder builder(atanOp);

      Attribute oneAttr0;
      if (isa<IntegerType>(ty.getElementType()))
        oneAttr0 = builder.getIntegerAttr(ty.getElementType(), 1);
      else if (isa<FloatType>(ty.getElementType()))
        oneAttr0 = builder.getFloatAttr(ty.getElementType(), 1);
      else if (auto CT = dyn_cast<ComplexType>(ty.getElementType()))
        oneAttr0 = complex::NumberAttr::get(CT, 1, 0);
      else
        return;

      DenseElementsAttr oneAttr;
      if (auto complexAttr = dyn_cast<complex::NumberAttr>(oneAttr0))
        oneAttr = DenseElementsAttr::get(ty, oneAttr0);
      else
        oneAttr = DenseElementsAttr::get(ty, oneAttr0);

      Value one =
          stablehlo::ConstantOp::create(builder, atanOp.getLoc(), oneAttr);
      Value res = stablehlo::Atan2Op::create(builder, atanOp.getLoc(),
                                             atanOp.getOperand(), one);
      atanOp.replaceAllUsesWith(res);
      atanOp.erase();
    });

    op->walk([=](arith::MaxNumFOp maxOp) {
      // maxnumf %a,%b -> select(isnan(%a), %b, max(%a, %b))
      if (!true || !isa<RankedTensorType>(maxOp.getResult().getType()))
        return;

      OpBuilder builder(maxOp);
      Value isLhsNaN =
          math::IsNaNOp::create(builder, maxOp.getLoc(), maxOp.getLhs());
      Value max = stablehlo::MaxOp::create(builder, maxOp.getLoc(),
                                           maxOp.getLhs(), maxOp.getRhs());
      Value res = stablehlo::SelectOp::create(builder, maxOp.getLoc(), isLhsNaN,
                                              maxOp.getRhs(), max);
      maxOp.replaceAllUsesWith(res);
      maxOp.erase();
    });
    op->walk([=](arith::MinNumFOp minOp) {
      // maxnumf %a,%b -> select(isnan(%a), %b, min(%a, %b))
      if (!true || !isa<RankedTensorType>(minOp.getResult().getType()))
        return;

      OpBuilder builder(minOp);
      Value isLhsNaN =
          math::IsNaNOp::create(builder, minOp.getLoc(), minOp.getLhs());
      Value min = stablehlo::MinOp::create(builder, minOp.getLoc(),
                                           minOp.getLhs(), minOp.getRhs());
      Value res = stablehlo::SelectOp::create(builder, minOp.getLoc(), isLhsNaN,
                                              minOp.getRhs(), min);
      minOp.replaceAllUsesWith(res);
      minOp.erase();
    });
    // op->walk([=](math::IsNaNOp nanOp) {
    //   if (!true || !isa<RankedTensorType>(nanOp.getResult().getType()))
    //     return;
    //
    //   OpBuilder builder(nanOp);
    //
    //   Value isFinite = stablehlo::IsFiniteOp::create(builder, nanOp.getLoc(),
    //                                                  nanOp.getOperand());
    //   Value isNotFinite =
    //       stablehlo::NotOp::create(builder, nanOp.getLoc(), isFinite);
    //
    //   Value isNotInf = stablehlo::NotOp::create(
    //       builder, nanOp.getLoc(),
    //       chlo::IsInfOp::create(builder, nanOp.getLoc(), nanOp.getOperand()));
    //
    //   Value isNaN = stablehlo::AndOp::create(builder, nanOp.getLoc(),
    //                                          isNotFinite, isNotInf);
    //
    //   nanOp.replaceAllUsesWith(isNaN);
    //   nanOp.erase();
    // });
    // op->walk([=](complex::ConjOp addOp) {
    //   if (!isa<RankedTensorType>(addOp->getResultTypes()[0]))
    //     return;
    //   OpBuilder builder(addOp);
    //   Value newAddOp;
    //   newAddOp =
    //       chlo::ConjOp::create(builder, addOp.getLoc(), addOp->getOperand(0));
    //   addOp.replaceAllUsesWith(newAddOp);
    //   addOp.erase();
    // });
    op->walk([=](arith::ConstantOp constOp) {
      if (!true || !isa<RankedTensorType>(constOp.getType()))
        return;

      auto valueAttr = constOp.getValueAttr();
      if (!isa<ElementsAttr>(valueAttr))
        return;

      OpBuilder builder(constOp);
      Value newConstOp =
          stablehlo::ConstantOp::create(builder, constOp.getLoc(), valueAttr);
      constOp.replaceAllUsesWith(newConstOp);
      constOp.erase();
    });
    op->walk([=](arith::FPToSIOp addOp) {
      if (!true || !isa<RankedTensorType>(addOp->getResultTypes()[0]))
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp = stablehlo::ConvertOp::create(
          builder, addOp.getLoc(), addOp->getOperand(0),
          cast<RankedTensorType>(addOp->getResult(0).getType())
              .getElementType());
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](arith::SIToFPOp addOp) {
      if (!true || !isa<RankedTensorType>(addOp->getResultTypes()[0]))
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp = stablehlo::ConvertOp::create(
          builder, addOp.getLoc(), addOp->getOperand(0),
          cast<RankedTensorType>(addOp->getResult(0).getType())
              .getElementType());
      if (cast<RankedTensorType>(addOp.getOperand().getType())
              .getElementType()
              .isInteger(1)) {
        newAddOp = stablehlo::NegOp::create(builder, addOp.getLoc(), newAddOp);
      }
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](arith::UIToFPOp addOp) {
      if (!true || !isa<RankedTensorType>(addOp->getResultTypes()[0]))
        return;
      if (!cast<RankedTensorType>(addOp.getOperand().getType())
               .getElementType()
               .isInteger(1)) {
        return;
      }
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp = stablehlo::ConvertOp::create(
          builder, addOp.getLoc(), addOp->getOperand(0),
          cast<RankedTensorType>(addOp->getResult(0).getType())
              .getElementType());
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](arith::ExtUIOp addOp) {
      if (!true || !isa<RankedTensorType>(addOp->getResultTypes()[0]))
        return;
      auto inTy =
          cast<RankedTensorType>(addOp.getOperand().getType()).getElementType();
      auto outTy = cast<RankedTensorType>(addOp.getType()).getElementType();
      bool legal = false;
      if (inTy.isInteger(1)) {
        legal = true;
      } else if (inTy.isInteger() && outTy.isInteger() &&
                 inTy.getIntOrFloatBitWidth() < outTy.getIntOrFloatBitWidth()) {
        legal = true;
      }
      if (!legal) {
        return;
      }
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp = stablehlo::ConvertOp::create(
          builder, addOp.getLoc(), addOp->getOperand(0),
          cast<RankedTensorType>(addOp->getResult(0).getType())
              .getElementType());
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    // op->walk([=](enzyme::BroadcastOp broadcastOp) {
    //   OpBuilder builder(broadcastOp);
    //   Value newBroadcastOp;
    //   assert(true);
    //   SmallVector<int64_t> broadcastDims;
    //   auto shape =
    //       cast<TensorType>(broadcastOp.getInput().getType()).getShape();
    //   broadcastDims.reserve(shape.size());
    //   for (auto en : llvm::enumerate(shape)) {
    //     // original dimensions end up one further because the batch dimension
    //     // is prepended:
    //     broadcastDims.push_back(en.index() + 1);
    //   }
    //   newBroadcastOp = stablehlo::BroadcastInDimOp::create(
    //       builder, broadcastOp.getLoc(), broadcastOp.getType(),
    //       broadcastOp.getInput(), builder.getDenseI64ArrayAttr(broadcastDims));
    //   broadcastOp.replaceAllUsesWith(newBroadcastOp);
    //   broadcastOp.erase();
    // });
    op->walk([=](arith::SelectOp selectOp) {
      if (!true ||
          llvm::any_of(selectOp->getOperandTypes(),
                       [](Type ty) { return !isa<RankedTensorType>(ty); }))
        return;

      OpBuilder builder(selectOp);
      auto newOp = stablehlo::SelectOp::create(
          builder, selectOp.getLoc(), selectOp.getType(),
          selectOp.getCondition(), selectOp.getTrueValue(),
          selectOp.getFalseValue());
      selectOp.replaceAllUsesWith(newOp.getResult());
      selectOp.erase();
    });
    op->walk([=](arith::CmpIOp cmpOp) {
      if (!isa<TensorType>(cmpOp.getType()))
        return;

      OpBuilder builder(cmpOp);

      Value newCmpOp;
      if (true) {
        stablehlo::ComparisonType compType = stablehlo::ComparisonType::SIGNED;
        auto predicate = cmpOp.getPredicate();
        if (predicate == arith::CmpIPredicate::ugt ||
            predicate == arith::CmpIPredicate::uge ||
            predicate == arith::CmpIPredicate::ult ||
            predicate == arith::CmpIPredicate::ule)
          compType = stablehlo::ComparisonType::UNSIGNED;

        stablehlo::ComparisonDirection direction;
        switch (predicate) {
        case arith::CmpIPredicate::eq:
          direction = stablehlo::ComparisonDirection::EQ;
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
        case arith::CmpIPredicate::ne:
          direction = stablehlo::ComparisonDirection::NE;
          break;
        default:
          return;
        }
        newCmpOp = stablehlo::CompareOp::create(
            builder, cmpOp->getLoc(), cmpOp->getOperand(0),
            cmpOp->getOperand(1), direction, compType);
      } else {
        return;
      }
      cmpOp.replaceAllUsesWith(newCmpOp);
      cmpOp.erase();
    });
    op->walk([=](arith::CmpFOp cmpOp) {
      if (!isa<TensorType>(cmpOp.getType()))
        return;

      // TODO: check fast math flags?
      OpBuilder builder(cmpOp);

      Value newCmpOp;
      stablehlo::ComparisonDirection direction;
      switch (cmpOp.getPredicate()) {
      case arith::CmpFPredicate::UEQ:
      case arith::CmpFPredicate::OEQ:
        direction = stablehlo::ComparisonDirection::EQ;
        break;
      case arith::CmpFPredicate::UGT:
      case arith::CmpFPredicate::OGT:
        direction = stablehlo::ComparisonDirection::GT;
        break;
      case arith::CmpFPredicate::UGE:
      case arith::CmpFPredicate::OGE:
        direction = stablehlo::ComparisonDirection::GE;
        break;
      case arith::CmpFPredicate::ULT:
      case arith::CmpFPredicate::OLT:
        direction = stablehlo::ComparisonDirection::LT;
        break;
      case arith::CmpFPredicate::ULE:
      case arith::CmpFPredicate::OLE:
        direction = stablehlo::ComparisonDirection::LE;
        break;
      case arith::CmpFPredicate::UNE:
      case arith::CmpFPredicate::ONE:
        direction = stablehlo::ComparisonDirection::NE;
        break;
      default:
        return;
      }
      newCmpOp = stablehlo::CompareOp::create(
          builder, cmpOp->getLoc(), cmpOp->getOperand(0),
          cmpOp->getOperand(1), direction, stablehlo::ComparisonType::FLOAT);
      
      cmpOp.replaceAllUsesWith(newCmpOp);
      cmpOp.erase();
    });
  }
};
} // end anonymous namespace

namespace xla_jit {
  std::unique_ptr<mlir::Pass> createArithRaisingPass() {
    return std::make_unique<ArithRaisingPass>();
  }   

  void registerArithRaisingPass() {
    ::mlir::registerPass([]()->std::unique_ptr<mlir::Pass>{return createArithRaisingPass();});
  }
}


