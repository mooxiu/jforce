#pragma once
#include <cstdint>
#include <vector>

// TODO: all the tests are using f32 for now.
enum class DType : int32_t {
  F32 = 0,
};

/**
 * Here, `rank` means the dimension of the data; shape is a vector of size of each dimension of the data.
 * For example, a <3x2 f32> matrix has rank = 2 and shape = {3, 2}.
*/
struct TensorDesc {
  void *data;
  int64_t *shape;
  int32_t rank;
  DType dtype;
};

struct KernelArgs {
  int32_t inputArgCount;
  TensorDesc *inputArgs;
  int32_t outputArgCount;
  TensorDesc *outputArgs;
};

// TODO: WTF is this function? Shouldn't `td.shape` can get the shape??????
std::vector<int64_t> getShape(TensorDesc td);
