#include "sharding-types.h"
#include <cassert>
#include <optional>

static bool sameAxis(const MeshAxis &a, const MeshAxis &b) {
  return a.name == b.name && a.size == b.size;
}

// Length of the longest common prefix of two axis lists.
static size_t commonPrefix(const DimSharding &a, const DimSharding &b) {
  size_t n = 0;
  while (n < a.size() && n < b.size() && sameAxis(a[n], b[n])) {
    ++n;
  }
  return n;
}

std::optional<int> compare(const DimSharding &dimSharding0,
                           const DimSharding &dimSharding1) {
  size_t prefix = commonPrefix(dimSharding0, dimSharding1);
  if (prefix == dimSharding0.size() && prefix == dimSharding1.size()) {
    return 0;
  }
  if (prefix == dimSharding0.size()) { // 0 is a prefix of 1: 0 is coarser
    return 1;
  }
  if (prefix == dimSharding1.size()) {
    return -1;
  }
  return std::nullopt;
}

DimSharding join(const DimSharding &dimSharding0,
                 const DimSharding &dimSharding1) {
  size_t prefix = commonPrefix(dimSharding0, dimSharding1);
  return DimSharding(dimSharding0.begin(), dimSharding0.begin() + prefix);
}

std::optional<int> compare(const TensorSharding &sharding0,
                           const TensorSharding &sharding1) {
  if (sharding0.size() != sharding1.size()) {
    return std::nullopt;
  }
  int result = 0;
  for (size_t dim = 0; dim < sharding0.size(); ++dim) {
    auto dimResult = compare(sharding0[dim], sharding1[dim]);
    if (!dimResult) {
      return std::nullopt;
    }
    if (*dimResult == 0) {
      continue;
    }
    // One dim coarser and another finer: not comparable.
    if (result != 0 && *dimResult != result) {
      return std::nullopt;
    }
    result = *dimResult;
  }
  return result;
}

TensorSharding join(const TensorSharding &sharding0,
                 const TensorSharding &sharding1) {
  assert(sharding0.size() == sharding1.size() &&
         "join needs shardings of the same rank");
  TensorSharding result;
  result.reserve(sharding0.size());
  for (size_t dim = 0; dim < sharding0.size(); ++dim) {
    result.push_back(join(sharding0[dim], sharding1[dim]));
  }
  return result;
}
