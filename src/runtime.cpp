#include <cstddef>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <regex>
#include <string>
#include <dlfcn.h>
#include "../third_party/headers/pjrt_c_api.h"

std::string getPluginPath() {
    // DEFAULT_PJRT_PLUGIN_PATH should be defined in CMake
    #ifdef DEFAULT_PJRT_PLUGIN_PATH
        return DEFAULT_PJRT_PLUGIN_PATH;
    #else
        throw std::runtime_error("PJRT plugin path not found. Please set PJRT_PLUGIN_PATH.");
    #endif
}

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
    
    static const std::regex addition_op_regexp(R"(stablehlo.add)");
    if (std::regex_match(stableHLO_func, addition_op_regexp)) {
        std::cout << "This is a addition operation" << std::endl;
    } else {
        std::cout << "other operations" << std::endl;
    }
}

void execute() {
    auto handle_ = dlopen(getPluginPath().c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
        std::cerr << "error loading plugin: " << dlerror() << std::endl;
        return;
    }

    void* get_api_sym = dlsym(handle_, "GetPjrtApi");
    if (!get_api_sym) {
        std::cerr << "error finding GetPjrtApi: " << dlerror() << std::endl;
        return;
    }

    // 3. 将符号转换为函数指针类型
    using GetPjrtApiFn = PJRT_Api* (*)();
    auto get_api_fn = reinterpret_cast<GetPjrtApiFn>(get_api_sym);
    
    auto api = get_api_fn();
    if (api) {
        std::cout << "[LOG] API loaded successfully" << std::endl;
        return;
    } 

    PJRT_Client_Create_Args args;
    PJRT_Client* client = nullptr;
    args.client = client;
    
    PJRT_Error* error = api->PJRT_Client_Create(&args);
    if (error) {
        std::cerr << "[Err] error creating client" << std::endl;
        return;
    }
    
    return;
}

extern "C" void launch_kernel(void* a_ptr, void* b_ptr, void* out_ptr, long n) {
    float* a_float_ptr = static_cast<float*>(a_ptr);
    float* b_float_ptr = static_cast<float*>(b_ptr);
    float* o_float_ptr = static_cast<float*>(out_ptr);

    /** 
        Start: testing Fortran could call runtime written in CPP
    */
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
    print_vec(a_float_ptr, n, "a_vec");
    print_vec(b_float_ptr, n, "b_vec");
    print_vec(o_float_ptr, n, "c_vec");
    /**
        End: testing Fortran could call runtime written in CPP
     */


    /**
        Start: Using real addition through XLA
     */
    execute();
    /**
        End: Using real addition through XLA
     */
}
