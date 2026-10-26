#ifndef SHARDING_TYPES_H
#define SHARDING_TYPES_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct MeshAxis {
  std::string name;
  int ordinal;
  int64_t size;

  MeshAxis(std::string name_, int ordinal_, int64_t size_)
      : name(name_), ordinal(ordinal_), size(size_) {};
};

// Mesh axes assigned to one tensor dimension; empty means replicated.
using DimSharding = std::vector<std::reference_wrapper<const MeshAxis>>;
// One complete tensor layout, indexed by tensor dimension.
using ArgSharding = std::vector<DimSharding>;
// One chosen layout per operand, result, or function argument.
using ShardingDecision = std::vector<ArgSharding>;

// One compatible combination of operand and result layouts for an operation.
// This is one strategy, not the Cartesian product of independent layouts.
// Internal communication and cost information will be added in the next step.
struct OperationShardingStrategy {
  ShardingDecision operands;
  ShardingDecision results;
};

// Boundary layouts of a kernel; outputs need not match inputs.
// Internal operation strategies will be recorded separately.
struct KernelShardingPlan {
  ShardingDecision inputs;
  ShardingDecision outputs;
};

#endif
