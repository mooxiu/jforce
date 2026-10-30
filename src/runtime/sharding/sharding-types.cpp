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

std::optional<int> compare(const DimSharding &dimPlacement0,
                           const DimSharding &dimPlacement1) {
  size_t prefix = commonPrefix(dimPlacement0, dimPlacement1);
  if (prefix == dimPlacement0.size() && prefix == dimPlacement1.size()) {
    return 0;
  }
  if (prefix == dimPlacement0.size()) { // 0 is a prefix of 1: 0 is coarser
    return 1;
  }
  if (prefix == dimPlacement1.size()) {
    return -1;
  }
  return std::nullopt;
}

DimSharding join(const DimSharding &dimPlacement0,
                 const DimSharding &dimPlacement1) {
  size_t prefix = commonPrefix(dimPlacement0, dimPlacement1);
  return DimSharding(dimPlacement0.begin(), dimPlacement0.begin() + prefix);
}

std::optional<int> compare(const ArgSharding &argPlacement0,
                           const ArgSharding &argPlacement1) {
  if (argPlacement0.size() != argPlacement1.size()) {
    return std::nullopt;
  }
  int result = 0;
  for (size_t dim = 0; dim < argPlacement0.size(); ++dim) {
    auto dimResult = compare(argPlacement0[dim], argPlacement1[dim]);
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

ArgSharding join(const ArgSharding &argPlacement0,
                 const ArgSharding &argPlacement1) {
  assert(argPlacement0.size() == argPlacement1.size() &&
         "join needs layouts of the same rank");
  ArgSharding result;
  result.reserve(argPlacement0.size());
  for (size_t dim = 0; dim < argPlacement0.size(); ++dim) {
    result.push_back(join(argPlacement0[dim], argPlacement1[dim]));
  }
  return result;
}
