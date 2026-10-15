#include "sharding-codec.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"


std::optional<std::reference_wrapper<const MeshAxis>>
getMeshAxis(const std::string &axisName, llvm::ArrayRef<MeshAxis> meshAxes) {
  for (const auto &meshAxis : meshAxes) {
    if (meshAxis.name == axisName) {
      return std::make_optional(meshAxis);
    }
  }
  return std::nullopt;
};

// {{{}, .., {}}, }
std::string toStr(const ShardingDecision &sd) {
  std::string res;
  res.push_back('{');
  for (const auto &argSd : sd) {
    res.push_back('{');
    for (const auto &dimSd : argSd) {
      res.push_back('{');
      for (const auto &axis : dimSd) {
        res += axis.get().name;
        res.push_back(',');
      }
      res.push_back('}');
    }
    res.push_back('}');
  }
  res.push_back('}');
  return res;
};

// Example: {{{axis-0,}{}{}}}
ShardingDecision toSD(const std::string &sdStr, llvm::ArrayRef<MeshAxis> meshAxes) {
  size_t pos = 0;
  auto expect = [&](char c) {
    if (pos >= sdStr.size() || sdStr[pos] != c) {
      llvm::report_fatal_error("Malformed sharding decision.");
    }
    pos += 1;
  };

  auto at = [&](char c) { return pos < sdStr.size() && sdStr[pos] == c; };

  ShardingDecision sd;
  expect('{');
  while (!at('}')) {

    ArgSharding argSd;
    expect('{');

    while (!at('}')) {

      DimSharding dimSd;
      expect('{');

      while (!at('}')) {
        const size_t start = pos;
        while (pos < sdStr.size() && sdStr[pos] != ',' && sdStr[pos] != '{' &&
               sdStr[pos] != '}') {
          pos += 1;
        }
        const std::string axisName = sdStr.substr(start, pos - start);
        expect(',');

        auto axis = getMeshAxis(axisName, meshAxes);
        if (!axis) {
          llvm::errs() << "Unknown mesh axis: " << axisName << "\n";
          llvm::report_fatal_error("Invalid sharding decision.");
        }
        dimSd.push_back(*axis);
      }
      expect('}');

      argSd.push_back(dimSd);
    }
    expect('}');
    sd.push_back(argSd);
  }
  expect('}');
  return sd;
};

