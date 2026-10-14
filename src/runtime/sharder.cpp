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
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <ios>
#include <istream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace nlohmann;


ShPGOProfile::ShPGOProfile(nlohmann::json &&data, Sharder &sharder) {
  if (!data.is_array()) {
    llvm::report_fatal_error("Sharding profile must be an array");
  }
  records.reserve(data.size());
    
  for (const auto& entry: data) {
    auto field = [&](const char* name) -> const nlohmann::json & {
      auto it = entry.find(name);
      if (it == entry.end()) {
        llvm::errs() << "Missing profile field: " << name << "\n";
      }
      return *it;
    };

    auto number = [&](const char* name) -> uint64_t {
      const auto& value = field(name);
      return value.get<uint64_t>();
    };

    auto text = [&](const char* name) -> std::string {
      const auto& value = field(name);
      return value.get<std::string>();
    };

    ShPGORecord rec{};
    rec.kernelFingerprint = static_cast<size_t>(number("kernelFingerprint"));
    rec.devMeshFingerprint = text("devMeshFingerprint");
    rec.hwFingerprint = text("hwFingerprint");
    rec.sd = sharder.toSD(text("sd"));
    rec.exeTime = number("exeTime");

    this->records.push_back(std::move(rec));
  }
}


// TODO: currently we only do 1D sharding, the deviceCount is automatically be
// comes the mesh
Sharder::Sharder(uint32_t deviceCount, std::string profilePath = "")
    : profile() {
  DEBUG_PRINT(
      llvm::formatv("Initializing Sharder. We got {0} devices.", deviceCount));
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
    this->profilePath = profilePath;
    this->deserializeProfile();
  }
}

Sharder::~Sharder() {
  if (!uncommitedRecords.empty()) {
    flushProfile();
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
std::string Sharder::toStr(const ShardingDecision &sd) {
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
ShardingDecision Sharder::toSD(const std::string &sdStr) {
  size_t pos = 0;
  auto expect = [&](char c) {
    if (pos >= sdStr.size() || sdStr[pos] != c) {
      llvm::report_fatal_error("Malformed sharding decision.");
    } 
    pos += 1;
  };

  auto at = [&](char c) {
    return pos < sdStr.size() && sdStr[pos] == c;
  };

  ShardingDecision sd;
  expect('{');
  while (!at('}')) {

    ArgSharding argSd;
    expect('{');

    while(!at('}')) {

      DimSharding dimSd;
      expect('{');

      while(!at('}')) {
        const size_t start = pos;
        while (pos < sdStr.size() && sdStr[pos] != ',' && sdStr[pos] != '{' && sdStr[pos] != '}') {
          pos += 1;
        }
        const std::string axisName = sdStr.substr(start, pos - start);
        expect(',');

        auto axis = getMeshAxis(axisName);
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

std::optional<std::reference_wrapper<const MeshAxis>>
Sharder::getMeshAxis(const std::string &axisName) {
  for (const auto &meshAxis : this->meshAxes) {
    if (meshAxis.name == axisName) {
      return std::make_optional(meshAxis);
    }
  }
  return std::nullopt;
};

void Sharder::flushProfile() {
  std::unique_lock<std::shared_mutex> wLock(profileMtx);
  if (!std::filesystem::exists(profilePath)) {
    llvm::errs() << "can not create corresponding profile file!\n";
    std::exit(EXIT_FAILURE);
  }
  std::ifstream file{profilePath};
  if (!file) {
    llvm::errs() << "cannot open profile: " << profilePath << "\n";
    std::exit(EXIT_FAILURE);
  }

  auto data = nlohmann::json::array();
  file >> std::ws;
  if (file.peek() != std::char_traits<char>::eof()) {
    file >> data;
  }
  for (const auto& entry: uncommitedRecords) {
    const auto& rec = entry.second;
    data.push_back({
      {"kernelFingerprint", rec.kernelFingerprint},
      {"devMeshFingerprint", rec.devMeshFingerprint},
      {"hwFingerprint", rec.hwFingerprint},
      {"sd", toStr(rec.sd)},
      {"exeTime", rec.exeTime},
    });
  }
  const std::filesystem::path tempPath = profilePath + ".tmp";
  {
    std::ofstream out(tempPath, std::ios::trunc);
    if (!out) {
      llvm::errs() << "Cannot create temporary profile\n";
      std::exit(EXIT_FAILURE);
    }
    out << data.dump(2);
    out.close();
  }
  std::filesystem::rename(tempPath, profilePath);
  uncommitedRecords.clear();
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
      file << "[]\n";
    }
  }

  std::shared_lock<std::shared_mutex> rLock(profileMtx);
  std::ifstream in(profilePath, std::ios_base::in);
  if (!in) {
    llvm::report_fatal_error("Cannot open sharding profile.");
  }
  nlohmann::json data = nlohmann::json::parse(in);
  if (data.is_discarded() || !data.is_array()) {
    llvm::errs() << "Data should be a valid json file!\n";
    std::exit(EXIT_FAILURE);
  }
  assert(data.is_array());
  this->profile = ShPGOProfile(std::move(data), *this);
  return;
}

uint64_t Sharder::getNextID() {
  return this->nextId_.fetch_add(1, std::memory_order_relaxed);
};

std::pair<uint64_t, std::chrono::time_point<std::chrono::steady_clock>>
Sharder::startRecordProfile(const ShardingDecision &sd,
                            size_t kernelFingerprint) {
  auto start = std::chrono::steady_clock::now();
  uint64_t id = getNextID();
  std::string devmeshFingerprint = "{";
  for (const auto &s : this->deviceMesh) {
    devmeshFingerprint += s;
    devmeshFingerprint.push_back(',');
  }
  devmeshFingerprint.push_back('}');
  ShPGORecord rec = {
      .kernelFingerprint = kernelFingerprint,
      .devMeshFingerprint = devmeshFingerprint,
      .hwFingerprint =
          "A100", // TODO: detect from the environment or by user input?
      .sd = sd,
  };
  {
    std::unique_lock<std::shared_mutex> lock(profileMtx);
    this->uncommitedRecords[id] = std::move(rec);
  }
  return std::make_pair(id, start);
};

void Sharder::endRecordProfile(uint64_t ID, const std::chrono::time_point<std::chrono::steady_clock>&& start) {
  auto end = std::chrono::steady_clock::now();

  std::unique_lock<std::shared_mutex> lock(profileMtx);
  auto it = this->uncommitedRecords.find(ID);
  if (it == uncommitedRecords.end()) {
    llvm::errs() << "Record not found in uncommitedRecords!\n";
    std::exit(EXIT_FAILURE);
  }
  auto& rec = it->second;
  rec.exeTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return;
}


