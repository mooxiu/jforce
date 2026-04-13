#include "kernel_pointer_interface.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

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
  }
}

std::vector<int64_t> getShape(TensorDesc td) {
  std::vector<int64_t> shape;
  shape.reserve(td.rank);
  for (int i = 0; i < td.rank; i++) {
    shape.push_back(td.shape[i]);
  }
  return shape;
}
