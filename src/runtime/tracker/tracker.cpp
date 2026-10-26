#include "tracker.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <functional>
#include <string>

Tracker::Tracker(const std::string& trackFilePath) {
};

Tracker::~Tracker(){
  
};


void Tracker::track(const std::string& originalKernelStr, func::FuncOp stableHLOKernel) {
  auto fingerPrint = std::hash<std::string>{}(originalKernelStr);
  trackFilePath.push_back(fingerPrint);
  auto it = kernels.find(fingerPrint);
  if (it == kernels.end()) {
    kernels[fingerPrint] = stableHLOKernel.clone();
  }
  return;
};

