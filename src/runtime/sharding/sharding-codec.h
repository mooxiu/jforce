#ifndef SHARDING_CODEC_H
#define SHARDING_CODEC_H

#include "mlir/IR/MLIRContext.h"
#include "sharding-types.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

std::string toStr(const ShardingDecision &sd);

ShardingDecision toSD(const std::string &sdStr,
                      llvm::ArrayRef<MeshAxis> meshAxes);

mlir::sdy::TensorShardingAttr toShardingAttr(mlir::MLIRContext &ctx,
                                             llvm::StringRef meshName,
                                             const ArgSharding &layout);

#endif
