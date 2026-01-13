#pragma once
#include <cstdint>
#include <string>
#include <vector>

// TODO: all the tests are using f32 for now.
enum class DType : int32_t {
  F32 = 0,
};

enum class TargetDevice : int32_t {
  CPU = 0,
  CUDA = 1,
};

/**
 *
 * data: Pointer to the beginning of the data,
 * shape: An array recording the size of each dimension of the data. For example, a <3x2 f32> matrix has shape = {3, 2},
 * rank: The dimension count of the data. For example, a <3x2 f32> matrix has rank = 2.
 * dtype: Data Type. An instance of enum DType.
 *
*/
struct TensorDesc {
  void *data;
  const int64_t *shape;
  int32_t rank;
  DType dtype;
};

struct KernelArgs {
  int32_t inputArgCount;
  TensorDesc *inputArgs;
  int32_t outputArgCount;
  TensorDesc *outputArgs;
  TargetDevice targetDevice;
};

// TODO: WTF is this function? Shouldn't `td.shape` can get the shape??????
std::vector<int64_t> getShape(TensorDesc td);
