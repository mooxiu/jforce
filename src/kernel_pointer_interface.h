#ifndef KERNELPOINTERINTERFACE_H
#define KERNELPOINTERINTERFACE_H

#include "../third_party/headers/pjrt_c_api.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

enum class DType : int32_t {
  F32 = 0,
  F64 = 1,
  I32 = 2,
  I64 = 3,
};

size_t getDTypeSizeInByte(DType dtype);
PJRT_Buffer_Type getPJRTBufferType(DType dtype);
DType getDTypeFromRankedTensorType(mlir::RankedTensorType tensorTy);

enum class TargetDevice : int32_t {
  CPU = 0,
  CUDA = 1,
  ROCM = 2,
  TPU = 3,
};

/**
 *
 * data: Pointer to the beginning of the data,
 * shape: An array recording the size of each dimension of the data. For
 * example, a <3x2 f32> matrix has shape = {3, 2}, rank: The dimension count of
 * the data. For example, a <3x2 f32> matrix has rank = 2. dtype: Data Type. An
 * instance of enum DType.
 *
 */
struct TensorDesc {
  void *data;
  const int64_t *shape;
  int32_t rank;
  DType dtype;
  bool isLiteral;

  // Return total elements of this tensor
  size_t getEleSize() {
    if (rank == 0) {
      return 1;
    }
    size_t acc = 1;
    for (int i = 0; i < rank; i++) {
      acc *= shape[i];
    }
    return acc;
  }

  void formatPrint() {
    llvm::dbgs() << "Data starts in: " << data << ", Shape: ";
    for (int i = 0; i < rank; i++) {
      llvm::dbgs() << " " << (int64_t)shape[i];
    }
    llvm::dbgs() << ", rank: " << rank << ", type: " << (int32_t)dtype << "\n";
  }
};

struct KernelArgs {
  unsigned int inputArgCount;
  TensorDesc *inputArgs;
  unsigned int outputArgCount;
  TensorDesc *outputArgs;
  TargetDevice targetDevice;

  void formatPrint() {
    llvm::dbgs() << "The Target Device is: " << (int32_t)targetDevice << "\n"
                 << "We have " << inputArgCount << " inputs: \n";
    for (int i = 0; i < inputArgCount; i++) {
      inputArgs[i].formatPrint();
    }
    llvm::dbgs() << "We have " << outputArgCount << " outputs: \n";
    for (int i = 0; i < outputArgCount; i++) {
      outputArgs[i].formatPrint();
    }
  }
};

#endif
