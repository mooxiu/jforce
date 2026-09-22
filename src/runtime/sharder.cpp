#include "sharder.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>


Sharder::Sharder(llvm::SmallVector<uint32_t> mesh) : deviceMesh(mesh) {
  if (std::any_of(mesh.begin(), mesh.end(), [](uint32_t i){return i == 0;})) {
    llvm::errs() << "Invalid mesh, should not have 0\n";
    std::exit(EXIT_FAILURE);
  }
  if (!profilePath.empty()) {
    this->profile = this->deserializeProfile();
  }
  this->deviceCount = std::reduce(mesh.begin(), mesh.end(), uint32_t(1), std::multiplies<>());
}

Sharder::~Sharder() {
  if (profileChangeFlag && !profilePath.empty()) {
    serializeProfile();
    return;
  }
}


llvm::SmallVector<ArgSharding> heuristicShard(mlir::ModuleOp moduleOp) {
  // TODO: implement this 
  



  // addToProfile
}


void Sharder::serializeProfile() {
  llvm_unreachable("serialize profile not omplemented");
}

Profile* Sharder::deserializeProfile() {
  llvm_unreachable("deserialize profile not implemented");
}


void addToProfile(ShardingDecision* decsion) {
  // TODO: maybe we can do topK etc.
  llvm_unreachable("add to profile not implemented");
}
