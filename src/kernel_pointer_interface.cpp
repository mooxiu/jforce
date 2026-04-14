#include "kernel_pointer_interface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdlib>
#include <iostream>

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

