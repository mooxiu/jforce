#ifndef SHARDING_TYPES_H
#define SHARDING_TYPES_H

#include <string>
#include <vector>

struct MeshAxis {
  std::string name;
  int ordinal;
  int64_t size;

  MeshAxis(std::string name_, int ordinal_, int64_t size_)
      : name(name_), ordinal(ordinal_), size(size_) {};
};

// Each arg is a nested dimenisonal arrary.
// ArgSharding[i] means how the i-th dimension of the arg sharded or replicaed.
// Because an argument has a
using DimSharding = std::vector<std::reference_wrapper<const MeshAxis>>;
using ArgSharding = std::vector<DimSharding>;
using ShardingDecision = std::vector<ArgSharding>;

#endif
