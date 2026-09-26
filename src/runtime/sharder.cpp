#include "sharder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>

Sharder::Sharder(llvm::SmallVector<uint32_t> mesh) : deviceMesh(mesh) {
  if (std::any_of(mesh.begin(), mesh.end(),
                  [](uint32_t i) { return i == 0; })) {
    llvm::errs() << "Invalid mesh, should not have 0\n";
    std::exit(EXIT_FAILURE);
  }
  if (!profilePath.empty()) {
    this->profile = this->deserializeProfile();
  }
  this->deviceCount =
      std::reduce(mesh.begin(), mesh.end(), uint32_t(1), std::multiplies<>());
}

Sharder::~Sharder() {
  if (profileChangeFlag && !profilePath.empty()) {
    serializeProfile();
    return;
  }
}

ShardingDecision heuristicShard(mlir::ModuleOp moduleOp) {
  // TODO: implement this
  auto kernelFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>("main");
  assert(kernelFunc && "sharder should be able to find the main function");
  for (unsigned int i = 0; i < kernelFunc.getNumArguments(); i++) {
    ::mlir::BlockArgument arg = kernelFunc.getArgument(i); 
    // kernelFunc.setArgAttr(i, "");

  }
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
