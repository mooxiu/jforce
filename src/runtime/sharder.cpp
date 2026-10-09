#include "jit-manager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "support/utilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <mutex>
#include <optional>
#include <ostream>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

using namespace nlohmann;



// TODO: currently we only do 1D sharding, the deviceCount is automatically be
// comes the mesh
Sharder::Sharder(uint32_t deviceCount, std::string profilePath = ""): profile() {
  DEBUG_PRINT(llvm::formatv("Initializing Sharder. We got {0} devices.", deviceCount));
  this->deviceCount = deviceCount;
  this->deviceMesh = {deviceCount};

  std::stringstream name;
  name << "dummyMesh";
  this->meshName = name.str();

  llvm::SmallVector<MeshAxis> meshAxes;
  meshAxes.reserve(deviceCount);
  for (int i = 0; i < deviceMesh.size(); i++) {
    meshAxes.emplace_back(llvm::formatv("axis-{0}", i), i, deviceMesh[i]);
  }
  this->meshAxes = std::move(meshAxes);

  if (std::any_of(deviceMesh.begin(), deviceMesh.end(),
                  [](uint32_t i) { return i == 0; })) {
    llvm::errs() << "Invalid mesh, should not have 0\n";
    std::exit(EXIT_FAILURE);
  }
  if (!profilePath.empty()) {
    this->deserializeProfile();
  }
}

Sharder::~Sharder() {
  if (profileChangeFlag && !profilePath.empty()) {
    serializeProfile();
    return;
  }
}

ShardingDecision Sharder::heuristicShard(mlir::ModuleOp moduleOp) {
  auto kernelFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>("main");
  assert(kernelFunc && "sharder should be able to find the main function");
  for (unsigned int i = 0; i < kernelFunc.getNumArguments(); i++) {
    ::mlir::BlockArgument arg = kernelFunc.getArgument(i);
    // kernelFunc.setArgAttr(i, "");
  }

  assert(kernelFunc.getNumArguments() == kernelFunc.getNumResults());
  std::vector<ArgSharding> argShardings;
  argShardings.resize(kernelFunc.getNumArguments());
  for (unsigned i = 0; i < kernelFunc.getNumArguments(); i++) {
    ArgSharding argSharding;
    auto arg = kernelFunc.getArgument(i);
    auto ty = llvm::dyn_cast<mlir::RankedTensorType>(arg.getType());
    if (ty.getRank() == 0) {
      // scale, just copy, set sharding as empty
      argShardings[i] = std::move(argSharding);
    } else {
      // Sharding dimension 0 in 0th-axis
      DimSharding dimSharding;
      dimSharding.push_back(meshAxes[0]);

      ArgSharding argSharding;
      argSharding.resize(ty.getRank());
      argSharding[0] = dimSharding;

      argShardings[i] = std::move(argSharding);
    }
  }
  return argShardings;

  // addToProfile
}

// {{{}, .., {}}, }
std::string Sharder::toStr(const ShardingDecision& sd) {
  std::string res;
  res.push_back('{');
  for (const auto& argSd: sd) {
    res.push_back('{');
    for (const auto& dimSd: argSd) {
      res.push_back('{'); 
        for (const auto& axis: dimSd) {
          res += axis.get().name;
          res.push_back(',');
        } 
        res.pop_back();
      res.push_back('}');
    } 
    res.push_back('}');
  }
  res.push_back('}');
  return res;
};

ShardingDecision Sharder::toSD(const std::string& sdStr) {
  ShardingDecision sd;
  int level = 0;
  for (int i = 0; i < sdStr.size(); i++) {
    if (sdStr[i] == '{') {
      level += 1;  
      if (level == 2) {
        ArgSharding argSd;
        while (!(sdStr[i] == '}' && level == 2)) {
          i += 1;
          DimSharding dimSd;
          if (sdStr[i] == '{') {
            level += 1;
          } else if (sdStr[i] == '}') {
            level -= 1;
          } else {
            std::string axisName;
            while (sdStr[i] != ',') {
              axisName.push_back(sdStr[i]);
              i += 1;
            }
            dimSd.push_back(getMeshAxis(axisName));
          }
          argSd.push_back(std::move(dimSd));
        }
        sd.push_back(std::move(argSd));
      }
    }
  } 
  return sd;
};

std::optional<MeshAxis&> Sharder::getMeshAxis(const std::string& axisName) {
  for (auto& meshAxis : this->meshAxes) {
    if (meshAxis.name == axisName) {
      return meshAxis;
    }
  }
  return std::nullopt;
};

void Sharder::serializeProfile() {
  std::unique_lock<std::shared_mutex> wLock(profileMtx);
  llvm_unreachable("serialize profile not omplemented");
}

void Sharder::deserializeProfile() {
  {
    std::unique_lock<std::shared_mutex> wLock(profileMtx);
    if (!std::filesystem::exists(profilePath)) {
      std::ofstream file{profilePath}; 
      if (!file) {
        llvm::errs() << "can not create corresponding profile file!\n";
        std::exit(EXIT_FAILURE);
      }
    }
  }

  std::shared_lock<std::shared_mutex> rLock(profileMtx);
  std::ifstream in(profilePath, std::ios_base::in);
  nlohmann::json data = nlohmann::json::parse(in);
  if (!data) {
    llvm::errs() << "parse fail!\n";
    std::exit(EXIT_FAILURE);
  }
  assert(data.is_array());
  this->profile = std::move(ShPGOProfile(std::move(data)));
  return;
}

void addToProfile(ShardingDecision *decsion) {
  // TODO: maybe we can do topK etc.
  llvm_unreachable("add to profile not implemented");
}
