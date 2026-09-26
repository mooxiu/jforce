#include "jit-manager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>

mlir::ModuleOp CacheManager::getModuleOp(void *JitCode) {
  auto JitCodePtr = reinterpret_cast<uintptr_t>(JitCode);
  char *JitCodeC = reinterpret_cast<char *>(JitCode);
  // If can found in map, just return a cloned moduleOP
  std::shared_lock<std::shared_mutex> rLock(moduleOpRWMtx);
  auto it = this->moduleOpMap.find(JitCodePtr);
  if (it != moduleOpMap.end()) {
    return it->getSecond()->clone();
  }
  rLock.unlock();

  // Else, need to parse
  mlir::ParserConfig parserConfig(&this->context_);
  mlir::OwningOpRef<mlir::ModuleOp> m =
      mlir::parseSourceString<mlir::ModuleOp>(JitCodeC, parserConfig);
  if (!m) {
    llvm::errs() << "Module not extracted!\n";
    exit(EXIT_FAILURE);
  }

  std::unique_lock<std::shared_mutex> wLock(moduleOpRWMtx);
  auto it2 = this->moduleOpMap.find(JitCodePtr);
  if (it2 != moduleOpMap.end()) {
    return it2->getSecond()->clone();
  }
  this->moduleOpMap[JitCodePtr] = std::move(m);
  return this->moduleOpMap[JitCodePtr]->clone();
}

// The cache key to the compiled kernel function.
// To uniquely identify a kernel function cache, we need to compare all the
// arguments which represents the shape.
//
// Key= JitCodePtr + [ArgSizes[i] + TgtArgs[i]] for i in NumAgrs
//
// JitCodePtr is the pointer to this JIT string captured.
//
// ArgSizes[i]:
// TgtArgs[i]:
llvm::SmallVector<uint64_t, 128> CacheManager::getL2JitMetasKey(
    int64_t NumArgs, void **TgtArgs, int64_t *ArgSizes, void *JitCode,
    const llvm::DenseMap<uint32_t, bool> &shapeArgInfoMap) {
  uintptr_t JitCodePtr = reinterpret_cast<std::uintptr_t>(JitCode);
  llvm::SmallVector<uint64_t, 128> key;

  key.push_back(JitCodePtr);
  for (int i = 0; i < NumArgs; i++) {
    key.push_back(ArgSizes[i]);
    assert(shapeArgInfoMap.contains(i));
    if (shapeArgInfoMap.at(i)) {
      key.push_back(reinterpret_cast<uintptr_t>(TgtArgs[i]));
    }
  }
  return key;
}

L1JitMetas *CacheManager::tryGetL1JitMetas(void *JitCode) {
  auto JitCodePtr = reinterpret_cast<uintptr_t>(JitCode);
  std::shared_lock<std::shared_mutex> rLock(this->l1JitMetaRWMtx);
  auto it = this->l1JitMetasMap.find(JitCodePtr);
  if (it != this->l1JitMetasMap.end()) {
    return &(it->getSecond());
  }
  return nullptr;
}

void CacheManager::saveL1JitMetas(
    void *JitCode, llvm::DenseMap<uint32_t, bool> shapeArgInfoMap) {
  uintptr_t JitCodePtr = reinterpret_cast<uintptr_t>(JitCode);
  std::unique_lock<std::shared_mutex> wLock(this->l1JitMetaRWMtx);
  if (!this->l1JitMetasMap.contains(JitCodePtr)) {
    this->l1JitMetasMap.try_emplace(
        JitCodePtr, L1JitMetas{.shapeArgInfoMap = std::move(shapeArgInfoMap)});
  }
  return;
}

L2JitMetas *
CacheManager::tryGetL2JitMetas(llvm::SmallVector<uint64_t, 128> &key) {
  std::shared_lock<std::shared_mutex> rLock(this->l2JitMetaRWMtx);
  auto it = this->l2JitMetasMap.find(key);
  if (it != l2JitMetasMap.end()) {
    return &(it->getSecond());
  }
  return nullptr;
}

L2JitMetas * CacheManager::insertL2CacheAndReturn(
    llvm::SmallVector<uint64_t, 128> &key, 
    PJRT_LoadedExecutable* exec, 
    std::string kernelFuncStr,
    mlir::func::FuncOp kernelFunc,
    std::function<void(PJRT_LoadedExecutable*)> destroyExec
    ) {
  std::unique_lock<std::shared_mutex> wLock(this->l2JitMetaRWMtx);
  auto it = this->l2JitMetasMap.find(key);
  if (it != l2JitMetasMap.end()) {
    // this->destroyLoadedExecutable(exec);
    destroyExec(exec);
    return &(it->getSecond());
  }
  

  auto funcTypes = kernelFunc.getFunctionType().getInputs();
  std::vector<mlir::Type> argTypesVec(funcTypes.begin(), funcTypes.end());


  auto insertedPair = this->l2JitMetasMap.try_emplace(
      key, (L2JitMetas){
               .exe = exec,
               .kernelFuncTypes = std::move(argTypesVec),
               .kernelFuncStr = std::move(kernelFuncStr),
           });
  return &(insertedPair.first->getSecond());
};
