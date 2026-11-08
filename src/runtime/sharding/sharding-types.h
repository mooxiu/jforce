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
// How one tensor is split over the mesh, indexed by tensor dimension.
// See TERMINOLOGY.md: a value has one value sharding (what devices hold); an
// operand sharding is what one use of it requires.
using TensorSharding = std::vector<DimSharding>;
// One sharding per operand, result, or kernel argument.
using TensorShardings = std::vector<TensorSharding>;


/// Sharding lattice: fewer axes is coarser, replicated {} is the top.
///   {x, y} < {x} < {}
/// a < b when b is a prefix of a; a becomes b by all-gathering the dropped axes.

// Partial order of DimShardings.
// return 1: dimSharding0 is coarser than dimSharding1
// return -1: dimSharding0 is finer than dimSharding1
// return 0: both shardings are the same
// nullopt: not comparable, e.g. {x} and {y}
std::optional<int> compare(
  const DimSharding& dimSharding0,
  const DimSharding& dimSharding1
);

// Finest sharding both can reach by coarsening: their longest common prefix.
// e.g. join({x, y}, {x}) = {x}, join({x}, {y}) = {}
DimSharding join(
  const DimSharding& dimSharding0,
  const DimSharding& dimSharding1
);

// Coarser if every dim is coarser or the same; nullopt if ranks differ.
std::optional<int> compare(
  const TensorSharding &sharding0,
  const TensorSharding &sharding1
);

// Join of each dim; both shardings must have the same rank.
TensorSharding join(
  const TensorSharding& sharding0,
  const TensorSharding& sharding1
);


// One candidate way to run an operation: the operand shardings it requires and
// the result shardings it produces (its communication cost is computed from
// these). This is one strategy, not the Cartesian product of shardings.
struct OpStrategy {
  TensorShardings operands;
  TensorShardings results;
};

// Shardings at the kernel boundary, indexed by argument: inputs[i] is how
// argument i is read in, outputs[i] how it is written back (result i is
// written back to argument i); outputs need not match inputs.
struct KernelIOShardings {
  TensorShardings inputs;
  TensorShardings outputs;
};

#endif
