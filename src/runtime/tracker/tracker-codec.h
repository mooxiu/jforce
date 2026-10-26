#ifndef TRACKER_CODEC_H
#define TRACKER_CODEC_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

std::string serializeFunction(func::FuncOp func);

#endif
