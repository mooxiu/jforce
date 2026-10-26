#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "runtime/sharding/sharding-codec.h"
#include "support/utilities.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace nlohmann;

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
  if (!uncommitedRecords_.empty()) {
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

  auto invocations = nlohmann::json::array();
  for (const auto &entry : uncommitedRecords_) {
    const auto &rec = entry.second;
    invocations.push_back({
        {"kernelFingerprint", rec.kernelFingerprint},
        {"argShapes", rec.argShapes},
        {"argDTypes", rec.argDTypes},
        {"sd", toStr(rec.sd)},
        {"kernelExeTime", rec.kernelExeTime},
    });
  }

  std::string devMeshFingerprint = "{";
  for (const auto &axis : this->deviceMesh) {
    devMeshFingerprint += std::to_string(axis);
    devMeshFingerprint.push_back(',');
  }
  devMeshFingerprint.push_back('}');

  data.push_back({
      {"devMeshFingerprint", devMeshFingerprint},
      {"hwFingerprint", "A100"}, // TODO: update this
      {"programExeTime", this->totalExecutionTime_},
      {"kernelRecords", invocations},
  });

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
  uncommitedRecords_.clear();
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
  this->profile = ShPGOProfile(std::move(data), this->meshAxes);
  return;
}

uint64_t Sharder::getNextID() {
  return this->nextId_.fetch_add(1, std::memory_order_relaxed);
};

std::pair<uint64_t, std::chrono::time_point<std::chrono::steady_clock>>
Sharder::startRecordProfile(const ShardingDecision &sd,
                            size_t kernelFingerprint, const KernelArgs &args) {
  auto start = std::chrono::steady_clock::now();
  uint64_t id = getNextID();

  std::vector<std::vector<int64_t>> argShapes;
  std::vector<DType> argDTypes;
  for (int idx = 0; idx < args.inputArgCount; idx++) {
    auto arg = args.inputArgs[idx];
    argShapes.push_back(std::move(arg.shape));
    argDTypes.push_back(std::move(arg.dtype));
  }

  std::string devmeshFingerprint = "{";
  for (const auto &s : this->deviceMesh) {
    devmeshFingerprint += s;
    devmeshFingerprint.push_back(',');
  }
  devmeshFingerprint.push_back('}');
  ShPGOKernelRecord rec = {
      .kernelFingerprint = kernelFingerprint,
      .argShapes = std::move(argShapes),
      .argDTypes = std::move(argDTypes),
      .sd = sd,
  };
  {
    std::unique_lock<std::shared_mutex> lock(profileMtx);
    this->uncommitedRecords_[id] = std::move(rec);
  }
  return std::make_pair(id, start);
};

void Sharder::endRecordProfile(
    uint64_t ID,
    const std::chrono::time_point<std::chrono::steady_clock> &&start) {
  auto end = std::chrono::steady_clock::now();

  std::unique_lock<std::shared_mutex> lock(profileMtx);
  auto it = this->uncommitedRecords_.find(ID);
  if (it == uncommitedRecords_.end()) {
    llvm::errs() << "Record not found in uncommitedRecords!\n";
    std::exit(EXIT_FAILURE);
  }
  auto &rec = it->second;
  rec.kernelExeTime =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  totalExecutionTime_ += rec.kernelExeTime;
  return;
}
