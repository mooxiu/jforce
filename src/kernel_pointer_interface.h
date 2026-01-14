#pragma once
#include <cstdint>
#include <vector>
#include <iostream>

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
  void formatPrint() {
    std::cout << "Data starts in: " << data;
    std::cout << ", Shape: " << shape;
    std::cout << ", rnak: " << rank;
    std::cout << ", type: " << (int32_t)dtype << std::endl;
  }
};

struct KernelArgs {
  int32_t inputArgCount;
  TensorDesc *inputArgs;
  int32_t outputArgCount;
  TensorDesc *outputArgs;
  TargetDevice targetDevice;

  void formatPrint() {
    std::cout << "The Target Device is: " << (int32_t)targetDevice << std::endl;
    std::cout << "We have " << inputArgCount << " inputs: \n";
    for (int i = 0; i < inputArgCount; i++) {
      inputArgs[i].formatPrint();
    }
    std::cout << "We have " << outputArgCount << " outputs: \n";
    for (int i = 0; i < outputArgCount; i++) {
      outputArgs[i].formatPrint();
    }
  }
};


// TODO: WTF is this function? Shouldn't `td.shape` can get the shape??????
std::vector<int64_t> getShape(TensorDesc td);
