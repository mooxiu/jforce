#include <cstdio>
#include <iostream>
#include <ostream>
#include <string>

// Source - https://stackoverflow.com/a
// Posted by Czarek Tomczak, modified by community. See post 'Timeline' for change history
// Retrieved 2025-11-24, License - CC BY-SA 3.0
auto replace_all= [](std::string& subject, const std::string& search, const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
};

auto addition_op_gen = [](int size) {
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
    std::cout << "addition op: " << op << std::endl;
    return op; 
};

void execute_stableHLO(std::string stableHLO_func, char** args) {
    std::cout << "stableHLO func: " << std::endl;
    std::cout << stableHLO_func << std::endl;
}

extern "C" void launch_kernel(void* a_ptr, void* b_ptr, void* out_ptr, long n) {
    float* a_float_ptr = static_cast<float*>(a_ptr);
    float* b_float_ptr = static_cast<float*>(b_ptr);
    float* o_float_ptr = static_cast<float*>(out_ptr);

    // 未来这里就是：StableHLO → PJRT → GPU
    for (long i = 0; i < n; ++i) {
        o_float_ptr[i] = a_float_ptr[i] + b_float_ptr[i];
    }

    auto print_vec = [](float* vec, int len, std::string name) {
        std::cout << "The element of " << name << ": ";
        for (int i = 0; i < len; i++) {
            std::cout << vec[i] << " ";
        } 
        std::cout << std::endl;
    };

    addition_op_gen(n);
    
    print_vec(a_float_ptr, n, "a_vec");
    print_vec(b_float_ptr, n, "b_vec");
    print_vec(o_float_ptr, n, "c_vec");
}
