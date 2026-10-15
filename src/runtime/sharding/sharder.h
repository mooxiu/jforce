#ifndef SHARDER_H
#define SHARDER_H

#include <atomic>
#include <cstdint>
#include "mlir/IR/BuiltinOps.h"
#include "sharding-types.h"
#include "sharding-profile.h"

class Sharder {
  friend class JitManager;

public:
  Sharder(uint32_t deviceSize, std::string profilePath);
  ~Sharder();
  Sharder(const Sharder &) = delete;
  Sharder &operator=(const Sharder &) = delete;

  // heuristicShard accept a moduleOp, and return a sharding decision.
  // The sharding decision is ideally made by using profilings.
  ShardingDecision heuristicShard(mlir::ModuleOp moduleOp);

  // Return the id for uncommited profiling record
  uint64_t getNextID();
  // Return <ID, start time point>
  std::pair<uint64_t, std::chrono::time_point<std::chrono::steady_clock>>
  startRecordProfile(const ShardingDecision &sd, size_t kernelFingerprint,
                     const KernelArgs &);
  void endRecordProfile(
      uint64_t ID,
      const std::chrono::time_point<std::chrono::steady_clock> &&start);

private:
  uint32_t deviceCount;
  std::string meshName;
  llvm::SmallVector<uint32_t> deviceMesh;
  llvm::SmallVector<MeshAxis> meshAxes;

  std::atomic<std::uint64_t> nextId_{1};
  uint64_t totalExecutionTime_; 
  llvm::DenseMap<uint64_t, ShPGOKernelRecord> uncommitedRecords_;

  // Profile parsed
  ShPGOProfile profile;
  std::string profilePath = "";
  // prevent two threads change the same profile file concurrently
  std::shared_mutex profileMtx;

  void flushProfile();
  void deserializeProfile();
};

#endif
