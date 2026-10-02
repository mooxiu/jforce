#include "jit-manager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// TODO: currently we only do 1D sharding, the deviceCount is automatically be
// comes the mesh
Sharder::Sharder(uint32_t deviceCount) {
  this->deviceCount = deviceCount;
  this->deviceMesh = {deviceCount};

  std::stringstream name;
  name << "dummyMesh";
  this->meshName = name.str();

  llvm::SmallVector<MeshAxis> meshAxes;
  meshAxes.resize(deviceCount);
  for (int i = 0; i < deviceMesh.size(); i++) {
    meshAxes[i] = MeshAxis{.name = llvm::formatv("axis-{0}", i),
                           .ordinal = i,
                           .size = deviceMesh[i]};
  }

  if (std::any_of(deviceMesh.begin(), deviceMesh.end(),
                  [](uint32_t i) { return i == 0; })) {
    llvm::errs() << "Invalid mesh, should not have 0\n";
    std::exit(EXIT_FAILURE);
  }
  if (!profilePath.empty()) {
    this->profile = this->deserializeProfile();
  }
}

Sharder::~Sharder() {
  if (profileChangeFlag && !profilePath.empty()) {
    serializeProfile();
    return;
  }
}

ShardingDecision Sharder::heuristicShard(mlir::ModuleOp moduleOp) {
  // TODO: implement this
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

void Sharder::serializeProfile() {
  llvm_unreachable("serialize profile not omplemented");
}

Profile *Sharder::deserializeProfile() {
  llvm_unreachable("deserialize profile not implemented");
}

void addToProfile(ShardingDecision *decsion) {
  // TODO: maybe we can do topK etc.
  llvm_unreachable("add to profile not implemented");
}
