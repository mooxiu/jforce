#ifndef SHARDING_CODEC_H
#define SHARDING_CODEC_H

#include "sharding-types.h"
#include "llvm/ADT/ArrayRef.h"

std::string toStr(const ShardingDecision &sd);
ShardingDecision toSD(const std::string &sdStr, llvm::ArrayRef<MeshAxis> meshAxes);

#endif
