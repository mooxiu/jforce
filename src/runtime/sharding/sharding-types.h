#ifndef SHARDING_TYPES_H
#define SHARDING_TYPES_H

#include <cstdint>
#include <functional>
#include <optional>
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


/// Sharding lattice: fewer axes is coarser, replicated {} is the top.
///   {x, y} < {x} < {}
/// a < b when b is a prefix of a; a becomes b by all-gathering the dropped axes.

// Partial order of DimShardings.
// return 1: dimPlacement0 is coarser than dimPlacement1
// return -1: dimPlacement0 is finer than dimPlacement1
// return 0: both placements are the same
// nullopt: not comparable, e.g. {x} and {y}
std::optional<int> compare(
  const DimSharding& dimPlacement0,
  const DimSharding& dimPlacement1
);

// Finest placement both can reach by coarsening: their longest common prefix.
// e.g. join({x, y}, {x}) = {x}, join({x}, {y}) = {}
DimSharding join(
  const DimSharding& dimPlacement0,
  const DimSharding& dimPlacement1
);

// Coarser if every dim is coarser or the same; nullopt if ranks differ.
std::optional<int> compare(
  const ArgSharding &argPlacement0,
  const ArgSharding &argPlacement1
);

// Join of each dim; both placements must have the same rank.
ArgSharding join(
  const ArgSharding& argPlacement0,
  const ArgSharding& argPlacement1
);


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
