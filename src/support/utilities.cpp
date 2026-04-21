#include "utilities.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdlib>
#include <iostream>

bool isLiteralTy(int64_t argType) { return (bool)(argType & 0x100); }

// FIXME: does not cover full situations, can be false negative.
bool isDynamicShape(mlir::Type type) {
  if (auto ref = llvm::dyn_cast<fir::ReferenceType>(type)) {
    return isDynamicShape(ref.getEleTy());
  }

  if (auto box = llvm::dyn_cast<fir::BoxType>(type)) {
    return isDynamicShape(box.getEleTy());
  }

  if (auto seq = llvm::dyn_cast<fir::SequenceType>(type)) {
    return seq.hasDynamicExtents();
  }

  if (auto exp = llvm::dyn_cast<hlfir::ExprType>(type)) {
    for (auto dim : exp.getShape()) {
      // TODO: is this correct usage?
      if (dim == mlir::ShapedType::kDynamic) {
        return true;
      }
    }
  }
  return false;
};

mlir::Type convertToStaticShape(mlir::Type type,
                                llvm::ArrayRef<int64_t> shape) {
  if (!isDynamicShape(type)) {
    return type;
  }

  return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
      .Case<fir::ReferenceType>([&](fir::ReferenceType rType) {
        return fir::ReferenceType::get(
            convertToStaticShape(rType.getEleTy(), shape));
      })
      .Case<fir::BoxType>([&](fir::BoxType bType) {
        // If the type is static, verifier won't accept box type anymore!
        // We need to replace it with a static ref.
        return fir::ReferenceType::get(convertToStaticShape(bType.getEleTy(), shape));
      })
      .Case<fir::SequenceType>([&](fir::SequenceType sType) {
        return fir::SequenceType::get(shape, sType.getEleTy());
      })
      .Case<hlfir::ExprType>([&](hlfir::ExprType eType) {
        return hlfir::ExprType::get(eType.getContext(), shape, eType.getEleTy(),
                                    eType.getPolymorphic());
      })
      .Default([&](mlir::Type t) { return t; });
};

std::string getMLIROperationAsString(mlir::Operation *op) {
  std::string output;
  llvm::raw_string_ostream os(output);
  mlir::OpPrintingFlags flags;
  op->print(os, flags.useLocalScope());
  return output;
}

DTypeVal extractLiteralPtr(uint64_t rawPtr, DType dType) {
  int32_t val_i32 = 0;
  int64_t val_i64 = 0;
  float val_f32 = 0.0;
  double val_f64 = 0.0;

  void *host_ptr = nullptr;

  switch (dType) {
  case DType::I32: {
    val_i32 = static_cast<int32_t>(rawPtr);
    host_ptr = &val_i32;
    break;
  }
  case DType::I64: {
    val_i64 = static_cast<int64_t>(rawPtr);
    host_ptr = &val_i64;
    break;
  }
  case DType::F32: {
    uint32_t low_bits = static_cast<uint32_t>(rawPtr);
    std::memcpy(&val_f32, &low_bits, sizeof(float));
    host_ptr = &val_f32;
    break;
  }
  case DType::F64: {
    std::memcpy(&val_f64, &rawPtr, sizeof(double));
    host_ptr = &val_f64;
    break;
  }
  }

  return DTypeVal {
    .valF32 = val_f32,
    .valF64 = val_f64,
    .valI32 = val_i32,
    .valI64 = val_i64,
    .returnedType = dType
  };
}

DType getDTypeFromRankedTensorType(mlir::RankedTensorType tensorTy) {
  auto eleType = tensorTy.getElementType();
  if (eleType.isF32()){
    return DType::F32;
  } else if (eleType.isF64()){
    return DType::F64;
  } else if (eleType.isInteger(32)) {
    return DType::I32;
  } else if (eleType.isInteger(64)) {
    return DType::I64;
  } else {
    llvm::errs() << "Unexpected ElementType of tensorTy: " << eleType << "\n";
    std::exit(EXIT_FAILURE);
  }
}

DType getDTypeFromValueType(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, DType>(type)
      .Case<fir::ReferenceType>([&](fir::ReferenceType rType) {
        return getDTypeFromValueType(rType.getEleTy());
      })
      .Case<fir::BoxType>([&](fir::BoxType bType) {
        return getDTypeFromValueType(bType.getEleTy());
      })
      .Case<fir::SequenceType>([&](fir::SequenceType sType) {
        return getDTypeFromValueType(sType.getEleTy());
      })
      .Case<hlfir::ExprType>([&](hlfir::ExprType eType) {
        return getDTypeFromValueType(eType.getEleTy());
      })
      .Default([&](mlir::Type t) { 
        if (t.isF32()) {
          return DType::F32;
        } else if (t.isF64()) {
          return DType::F64;
        } else if (t.isInteger(32)) {
          return DType::I32;
        } else if (t.isInteger(64)) {
          return DType::I64;
        }
          llvm::errs() << "Unexpected Type!\n";
          std::exit(EXIT_FAILURE);
      });
}

PJRT_Buffer_Type getPJRTBufferType(DType dtype) {
  switch (dtype) {
  case DType::F32:
    return PJRT_Buffer_Type::PJRT_Buffer_Type_F32;
  case DType::F64:
    return PJRT_Buffer_Type::PJRT_Buffer_Type_F64;
  case DType::I32:
    return PJRT_Buffer_Type::PJRT_Buffer_Type_S32;
  case DType::I64:
    return PJRT_Buffer_Type::PJRT_Buffer_Type_S64;
  default:
    std::cerr << "Unsupported Type!\n";
    std::exit(EXIT_FAILURE);
  }
}

size_t getDTypeSizeInByte(DType dtype) {
  switch (dtype) {
  case DType::F32:
  case DType::I32:
    return size_t(4);
  case DType::F64:
  case DType::I64:
    return size_t(8);
  default:
    llvm::errs() << "Unexpected DType!\n";
    std::exit(EXIT_FAILURE);
  }
}

