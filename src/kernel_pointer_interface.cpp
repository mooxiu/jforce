#include "kernel_pointer_interface.h"
#include <vector>

std::vector<long> getShape(TensorDesc td) {
  std::vector<long> shape;
  shape.reserve(td.rank);
  for (int i = 0; i < td.rank; i++) {
    shape.push_back(td.shape[i]);
  }
  return shape;
}
