#ifndef SHARDER_H
#define SHARDER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <cstdlib>
#include <sys/types.h>

// Sharding Strategy is what we would like to partition an argument
// NOTE: sharding is in GSPMD style.
// For example, we have a 2x2 mesh.
// {0, 0} means 
struct ArgSharding {
  // sharding is in GSMPD Style
  llvm::SmallVector<uint32_t> sharding;
};

// Keeps the sharding of all arguments of a function.
struct ShardingDecision {
  void* jitCodePtr;
  llvm::SmallVector<ArgSharding> argShardings;
};

struct Profile {
  llvm::DenseMap<void*, ShardingDecision> decisions;  
};

class Sharder {
  public:
    Sharder(llvm::SmallVector<uint32_t> mesh);
    ~Sharder();
    Sharder(const Sharder&) = delete;
    Sharder& operator=(const Sharder&) = delete;
   
    ShardingDecision heuristicShard(mlir::ModuleOp moduleOp);

  private:
    uint32_t deviceCount;
    llvm::SmallVector<uint32_t> deviceMesh;

    Profile* profile = nullptr;
    bool profileChangeFlag = false;
    std::string profilePath = "";
    
    void addToProfile();
    void serializeProfile();
    Profile* deserializeProfile();
};

#endif
