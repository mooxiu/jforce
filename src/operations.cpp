#include "operations.h"
#include <iostream>


#define SPH "{SHAPE}" // Shape Place Holder

void logger::Log(std::string msg, logLevel level) {
  switch (level) {
    case logLevel::DEBUG:
      std::cout << "[DEBUG] " << msg << std::endl;
      break;
    case logLevel::ERROR:
      std::cerr << "[ERROR] " << msg << std::endl;
      break;
  }
}


// Source - https://stackoverflow.com/a
// Posted by Czarek Tomczak, modified by community. See post 'Timeline' for change history
// Retrieved 2025-11-24, License - CC BY-SA 3.0
void replace_all(std::string& subject, const std::string& search, const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
};


std::string GetVectorAdditionOp(int size) {
    // StableHLO
    std::string op =
        R"(
            module @jit_addition attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} { 
                func.func public @main(%arg0: tensor<{SHAPE}xf32>, %arg1: tensor<{SHAPE}xf32>) -> (tensor<{SHAPE}xf32> {jax.result_info = "result"}) {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<{SHAPE}xf32> 
                    return %0 : tensor<{SHAPE}xf32> 
                } 
            }
        )";

    replace_all(op, SPH, std::to_string(size));
    logger::Log("addition op: "+ op, logLevel::DEBUG);
    return op; 
};