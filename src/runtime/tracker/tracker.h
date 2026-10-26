#ifndef TRACKER_H
#define TRACKER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <cstdint>
#include <sys/types.h>


using namespace mlir;

/// Tracker record the stablehlo functions and their order.
/// One tracker one execution, data can be appended in a track file.
///
/// [hash0: funcOp, hash1: funcOp, ...]
class Tracker {
  private:
    //TODO: add a mutex?
  
    // key: the hash of the original non-specialized function. -> for indentification.
    // value: the specialized StableHLO function. -> for estimating communication cost.
    //
    // Notice that even though the function is size specialized, if the key is the same, different sizes should still use the same StableHLO function.
    // Because when estimating the cost, we do not care the real size.
    // Also, because based on current JForce lowering pipeline, it is difficult to get the StableHLO function without size specialization.
    llvm::DenseMap<size_t, func::FuncOp> kernels;
    llvm::SmallVector<uint64_t> executionTrack;
    std::string trackFilePath;
    Tracker(const std::string& trackFilePath);
    ~Tracker();
  public:
    void track(const std::string& originalKernelStr, func::FuncOp stableHLOKernel);
};

#endif
