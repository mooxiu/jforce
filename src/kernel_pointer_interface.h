#pragma once
#include <cstdint>
#include <vector>

enum class OpType : int32_t {
    VECTOR_ADD = 0,
    DOT_PRODUCT = 1,
    TRANSPOSE = 2,
    MATRIX_MUL = 3
};

// TODO: all the tests are using f32 for now
enum class DType: int32_t {
    F32 = 0,
};

struct TensorDesc {
    void* data;
    int64_t* shape;
    int32_t rank;
    DType dtype;
};

struct KernelArgs {
    OpType opCode;
    long inputArgCount;
    TensorDesc* inputArgs;
    long outputArgCount;
    TensorDesc* outputArgs;
};

std::vector<long> getShape(TensorDesc td);