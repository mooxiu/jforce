#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <ostream>
#include <regex>
#include <string>
#include <dlfcn.h>
#include "../third_party/headers/pjrt_c_api.h"
#include "../third_party/protos/generated/xla/pjrt/proto/compile_options.pb.h"

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
    

    // HLO
    std::string op_bk = 
        R"(
            HloModule jit_addition, entry_computation_layout={(f32[{SHAPE}]{0}, f32[{SHAPE}]{0})->f32[{SHAPE}]{0}}

                ENTRY main.1 {
                    x.1 = f32[{SHAPE}]{0} parameter(0)
                    y.1 = f32[{SHAPE}]{0} parameter(1)
                    ROOT add.1 = f32[{SHAPE}]{0} add(x.1, y.1)
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

std::string get_err_msg(PJRT_Api* api, PJRT_Error* err) {
    PJRT_Error_GetCode_Args code_args = {};
    code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    code_args.error = err;

    api->PJRT_Error_GetCode(&code_args);

    PJRT_Error_Message_Args msg_args = {};
    msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    msg_args.error = err;
    api->PJRT_Error_Message(&msg_args); 
    std::string s (msg_args.message);

    PJRT_Error_Destroy_Args destroy_args = {};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = err;

    api->PJRT_Error_Destroy(&destroy_args);
    return s;
}

void execute(std::string func_code) {
    auto handle_ = dlopen(getPluginPath().c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
        std::cerr << "error loading plugin: " << dlerror() << std::endl;
        return;
    }
    // follow the example of `man dlopen`
    auto get_api_fn = (PJRT_Api* (*)())dlsym(handle_, "GetPjrtApi");
    if (!get_api_fn) {
        std::cerr << "error finding GetPjrtApi: " << dlerror() << std::endl;
        return;
    }
    auto api = get_api_fn();
    std::cout << "[LOG] the api loaded successfully!" << std::endl;

    /**
    Get the client
     */
    PJRT_Client_Create_Args args = {};
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* error = api->PJRT_Client_Create(&args);
    if (error) {
        std::cerr << "[Err] error creating client" << std::endl;
        return;
    }
    std::cout << "[LOG] client is successfully created" << std::endl;

    
    /**
    Compile the stableHLO
     */
    // TODO: why not use `PJRT_Compile` rather than `PJRT_Client_Compile`?
    PJRT_Client_Compile_Args compile_args = {};
    compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;

    PJRT_Program program = {};
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    std::string format = "mlir";
    // program.code = (char *) func_code.c_str();
    // program.code_size = func_code.size();
    std::cout << "[DEBUG] The code is : " << func_code << std::endl; 
    // We have to set as mlir here as we're passing MLIR module string rather than serialized HLOModuleProto
    program.code = (char*) func_code.c_str();
    program.code_size = (size_t)func_code.size();
    program.format = format.c_str();
    program.format_size = (size_t)format.size();

    compile_args.client = args.client;
    compile_args.program = &program;
    xla::CompileOptionsProto opts = {};
    opts.set_parameter_is_tupled_arguments(false);
    opts.set_compile_portable_executable(false);
    opts.set_profile_version(1);

    // 设置 num_replicas 等
    xla::ExecutableBuildOptionsProto* build_opts =
        opts.mutable_executable_build_options();
    build_opts->set_num_replicas(1);
    build_opts->set_num_partitions(1);

    // 序列化
    std::string buf;
    if (!opts.SerializeToString(&buf)) {
        // 这里你自己决定怎么报错
        throw std::runtime_error("failed to serialize CompileOptionsProto");
    }
    compile_args.compile_options = (char *)buf.c_str();
    compile_args.compile_options_size = (size_t)buf.size();
    

    error = api->PJRT_Client_Compile(&compile_args);
    if (error) {
        std::cerr << "[ERR] error compiling the program: " << get_err_msg(api, error) << std::endl;
        return;
    }
    std::cout << "[LOG] program is compiled successfully" << std::endl;

    /**
    Dealing with the buffer
     */

    /**
    Execute the program
     */
    PJRT_LoadedExecutable* exe = compile_args.executable;
    if (!exe) {
        std::cerr << "[ERR] compile shows no error, but no exe produced" << std::endl; 
        return;
    }
    std::cout << "[LOG] got the exe" << std::endl;

    PJRT_LoadedExecutable_Execute_Args leeas;
    leeas.executable = exe;
    error = api->PJRT_LoadedExecutable_Execute(&leeas);
    if (error) {
        std::cerr << "[ERR] fail to execute" << std::endl;
        return;
    }
    std::cout << "[LOG] execute successfully" << std::endl;


    // TODO: which one should I use? PJRT_LoadedExecutable_Delete or this?
    PJRT_LoadedExecutable_Destroy_Args ledargs;
    ledargs.executable = exe; 
    error = api->PJRT_LoadedExecutable_Destroy(&ledargs);
    if (error) {
        // TODO: 
        return;
    }

    dlclose(handle_);

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
    auto op = addition_op_gen(8);
    execute(op);
    /**
        End: Using real addition through XLA
     */
}
