#include "llvm/ADT/DenseMap.h"
#include "../third_party/headers/pjrt_c_api.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstdint>
#include <functional>
#include <string>

using namespace mlir;

#define DEFAULT_VOLUME 16

class kernelCache {
  llvm::DenseMap<mlir::func::FuncOp, PJRT_Executable*> executableMaps;
  uint32_t volume = DEFAULT_VOLUME;  

  public:
    PJRT_Executable* find(mlir::func::FuncOp funcOp) {
      if (!executableMaps.contains(funcOp)) {
        return nullptr;  
      };
      return executableMaps.at(funcOp);
    };

    kernelCache(){};
    kernelCache(uint32_t v): volume(v) {}; 
};


