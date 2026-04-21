#ifndef UTILITIES_H
#define UTILITIES_H

#include "mlir/IR/Types.h"
#include <cmath>
#include <cstdint>
#include "../../third_party/headers/pjrt_c_api.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

#ifdef ENABLE_XLA_DEBUG
#define SET_XLA_FLAG()                                                         \
  setenv("XLA_FLAGS", "--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text", 1)
#define DEBUG_PRINT(str) llvm::dbgs() << "[DEBUG]" << str << "\n"
#else
#define SET_XLA_FLAG()
#define DEBUG_PRINT(str)
#endif


bool isLiteralTy(int64_t argType);

bool isDynamicShape(mlir::Type type);

mlir::Type convertToStaticShape(mlir::Type type, llvm::ArrayRef<int64_t> shape);

std::string getMLIROperationAsString(mlir::Operation *op);

enum class DType : int32_t {
  F32 = 0,
  F64 = 1,
  I32 = 2,
  I64 = 3,
};

size_t getDTypeSizeInByte(DType dtype);

PJRT_Buffer_Type getPJRTBufferType(DType dtype);

DType getDTypeFromRankedTensorType(mlir::RankedTensorType tensorTy);

DType getDTypeFromValueType(mlir::Type arg);

// This could be achieved by elegant sum type std::variant<> in Cpp, but it requires C++17 and pattern matching part is pretty ugly in Cpp.
struct DTypeVal {
  float_t valF32;
  double_t valF64;
  int32_t valI32;
  int64_t valI64;
  DType returnedType;
};

DTypeVal extractLiteralPtr(uint64_t rawPtr, DType dType);

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
