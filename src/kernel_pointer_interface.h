#pragma once
#include <cstdint>
#include <vector>

// TODO: all the tests are using f32 for now
enum class DType : int32_t {
  F32 = 0,
};

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

std::vector<int64_t> getShape(TensorDesc td);
