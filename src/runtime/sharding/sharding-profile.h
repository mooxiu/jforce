#ifndef SHARDING_PROFILE_H
#define SHARDING_PROFILE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "nlohmann/json_fwd.hpp"
#include "sharding-types.h"
#include "support/utilities.h"

class Sharder;

// Profile to be serialized or deserialized, inserted.
// {
//  "devMeshFingerprint": "...",
//  "hwFingerprint": "...",
//  "programExeTime": 123456,
//  {
///   "kernelFingerprint": 1234,
//    "argShapes": [[1024, 64], [64], []],
//    "argDTypes": ["f32", "f32", "i32"],
//    "sd": "{{{axis-0,}{}}{{}}{}}",
//    "kernelExeTime": 123456
//  },
//  {
//    ...
//  },,
// }
struct ShPGOKernelRecord {
  size_t kernelFingerprint;
  std::vector<std::vector<int64_t>> argShapes;
  std::vector<DType> argDTypes;
  ShardingDecision sd;
  uint64_t kernelExeTime;
};

struct ShPGOProgramRecord {
  std::string devMeshFingerprint;
  std::string hwFingerprint;
  std::vector<ShPGOKernelRecord> kernelRecords;
};

struct ShPGOProfile {
  llvm::SmallVector<ShPGOProgramRecord> records;
  ShPGOProfile() = default;
  ShPGOProfile(nlohmann::json &&data, llvm::ArrayRef<MeshAxis> meshAxes);
};

#endif
