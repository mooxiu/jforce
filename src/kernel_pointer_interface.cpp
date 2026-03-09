#include "kernel_pointer_interface.h"
#include <cstddef>
#include <cstdint>
#include <vector>

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
