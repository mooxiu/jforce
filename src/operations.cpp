#include "operations.h"
#include <cstdlib>
#include <iostream>


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

    replace_all(op, "{SHAPE}", std::to_string(size));
    logger::Log("addition op: "+ op, logLevel::DEBUG);
    return op; 
};

std::string GetDotProductOp(int size) {
    std::string op = 
        R"(
            module @jit_dot_product attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
                func.func public @main(%arg0: tensor<{SHAPE}xf32>, %arg1: tensor<{SHAPE}xf32>) -> (tensor<f32> {jax.result_info = "result"}) {
                    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) -> tensor<f32>
                    return %0 : tensor<f32>
                }
            }
        )";
    replace_all(op, "{SHAPE}", std::to_string(size));
    logger::Log("dot product op: "+ op, logLevel::DEBUG);
    return op;
}

std::string GetTransposeOp(std::vector<int> shape) {
    std::string op = 
        R"(
            module @jit_transpose attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
                func.func public @main(%arg0: tensor<{SHAPE1}xf32>) -> (tensor<{SHAPE2}xf32> {jax.result_info = "result"}) {
                    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<{SHAPE1}xf32>) -> tensor<{SHAPE2}xf32>
                    return %0 : tensor<{SHAPE2}xf32>
                }
            }
        )";
    
    replace_all(op, "{SHAPE1}", std::to_string(shape[0]) + "x" + std::to_string(shape[1]));
    replace_all(op, "{SHAPE2}", std::to_string(shape[1]) + "x" + std::to_string(shape[0]));
    logger::Log("transpose op: "+ op, logLevel::DEBUG);
    return op;
}

std::string GetMatrixMultiplicationOp(std::vector<int> m1Shape, std::vector<int> m2Shape) {
    // Sanity check
    if (m1Shape[1] != m2Shape[0]) {
        logger::Log("Shape not compatible!", logLevel::ERROR);
        exit(EXIT_FAILURE);
    }
    std::string op = 
        R"(
            module @jit_matrix_multiply attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
                func.func public @main(%arg0: tensor<{SHAPE1}xf32>, %arg1: tensor<{SHAPE2}xf32>) -> (tensor<{SHAPE3}xf32> {jax.result_info = "result"}) {
                    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<{SHAPE1}xf32>, tensor<{SHAPE2}xf32>) -> tensor<{SHAPE3}xf32>
                    return %0 : tensor<{SHAPE3}xf32>
                }
            }            
        )";
    std::string shape1 = std::to_string(m1Shape[0]) + "x" + std::to_string(m1Shape[1]);
    std::string shape2 = std::to_string(m2Shape[0]) + "x" + std::to_string(m2Shape[1]);
    std::string shape3 = std::to_string(m1Shape[0]) + "x" + std::to_string(m2Shape[1]);
    replace_all(op, "{SHAPE1}", shape1);
    replace_all(op, "{SHAPE2}", shape2);
    replace_all(op, "{SHAPE3}", shape3);
    return op;
}