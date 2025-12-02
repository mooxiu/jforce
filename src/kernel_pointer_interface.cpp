#include "kernel_pointer_interface.h"
#include <cstdint>
#include <vector>

std::vector<int64_t> getShape(TensorDesc td) {
  std::vector<int64_t> shape;
  shape.reserve(td.rank);
  for (int i = 0; i < td.rank; i++) {
    shape.push_back(td.shape[i]);
  }
  return shape;
}
