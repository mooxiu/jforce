#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <dlfcn.h>
#include "../third_party/headers/pjrt_c_api.h"
#include "../third_party/protos/generated/xla/pjrt/proto/compile_options.pb.h"
#include "xla/xla.pb.h"

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

// Just for testing
// void execute_stableHLO(std::string stableHLO_func, char** args) {
    
//     static const std::regex addition_op_regexp(R"(stablehlo.add)");
//     if (std::regex_match(stableHLO_func, addition_op_regexp)) {
//         std::cout << "This is a addition operation" << std::endl;
//     } else {
//         std::cout << "other operations" << std::endl;
//     }
// }

std::string get_err_msg(const PJRT_Api* api, PJRT_Error* err) {
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

PJRT_Api* extract_api() {
     auto handle_ = dlopen(getPluginPath().c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
        std::cerr << "error loading plugin: " << dlerror() << std::endl;
        return nullptr;
    }
    // follow the example of `man dlopen`
    auto get_api_fn = (PJRT_Api* (*)())dlsym(handle_, "GetPjrtApi");
    if (!get_api_fn) {
        std::cerr << "error finding GetPjrtApi: " << dlerror() << std::endl;
        return nullptr;
    }
    auto api = get_api_fn();
    std::cout << "[LOG] the api loaded successfully!" << std::endl;
    return api;
}

PJRT_Client* create_client(const PJRT_Api* api) {
    PJRT_Client_Create_Args args = {};
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* error = api->PJRT_Client_Create(&args);
    if (error) {
        std::cerr << "[Err] error creating client" << std::endl;
        return nullptr;
    }
    std::cout << "[LOG] client is successfully created" << std::endl;
    return args.client; 
}

PJRT_LoadedExecutable* compile_mlir(
    const PJRT_Api* api, PJRT_Client* client, const std::string& func_code) {
    // TODO: `PJRT_Compile` or `PJRT_Client_Compile`?
    // It seems PJRT_Client_Compile will also help to load the execute
    PJRT_Client_Compile_Args compile_args = {};
    compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;

    PJRT_Program program = {};
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    // We have to set as mlir here as we're passing MLIR module string rather than serialized HLOModuleProto.
    std::string format = "mlir";
    std::cout << "[DEBUG] The code is : " << func_code << std::endl; 
    program.code = (char*) func_code.c_str();
    program.code_size = (size_t)func_code.size();
    program.format = format.c_str();
    program.format_size = (size_t)format.size();

    compile_args.client = client;
    compile_args.program = &program;
    xla::CompileOptionsProto opts = {};
    opts.set_parameter_is_tupled_arguments(false);
    opts.set_compile_portable_executable(false);
    opts.set_profile_version(1);

    xla::ExecutableBuildOptionsProto* build_opts =
        opts.mutable_executable_build_options();
    build_opts->set_num_replicas(1);
    build_opts->set_num_partitions(1);

    std::string buf;
    // SerializeToString(): This is protobuf's method inherited by CompileOptionProto.
    if (!opts.SerializeToString(&buf)) {
        std::cerr << "Faile to serialize CompileOptionsProto" << std::endl;
        return nullptr;
    }
    compile_args.compile_options = (char *)buf.c_str();
    compile_args.compile_options_size = (size_t)buf.size();
    

    PJRT_Error* error;
    error = api->PJRT_Client_Compile(&compile_args);
    if (error) {
        std::cerr << "[ERR] error compiling the program: " << get_err_msg(api, error) << std::endl;
        return nullptr;
    }
    std::cout << "[LOG] program is compiled successfully" << std::endl;
    return compile_args.executable;
}

// TODO: if we want to implement auto-sharding, we may need multiple devices
PJRT_Device* findCPUDevice(PJRT_Api* api, PJRT_Client* client) {
    PJRT_Client_AddressableDevices_Args device_args = {};
    device_args.struct_size = PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE;
    device_args.client = client;
    auto err = api->PJRT_Client_AddressableDevices(&device_args);
    if (err) {
        std::cerr << "[ERR] fail to find any device: " << get_err_msg(api, err) << std::endl;
        return nullptr;
    }
    if (device_args.num_addressable_devices < 1) {
        std::cerr << "[ERR] cannot find any device!" << std::endl;
        return nullptr;
    }
    std::cout << "[LOG] got " << device_args.num_addressable_devices << " devices." << std::endl;

    auto get_description = [api](PJRT_Device* device)-> std::string {
        PJRT_Device_GetDescription_Args args = {};
        args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
        args.device = device;
        auto err1 = api->PJRT_Device_GetDescription(&args);
        if (err1) {
            std::cerr << "[ERR] fail to get description of device: " << get_err_msg(api, err1) << std::endl;
            return nullptr;
        }
        PJRT_DeviceDescription_ToString_Args ts_args = {};
        ts_args.struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
        ts_args.device_description = args.device_description;
        auto err2 = api->PJRT_DeviceDescription_ToString(&ts_args);
        if (err2) {
            std::cerr << "[ERR] fail to get device description to string: " << get_err_msg(api, err2) << std::endl;
            return nullptr;
        }
        return ts_args.to_string;
    };

    int chosen_device_idx = 0;
    for (int i = 0; i < device_args.num_addressable_devices; i++) {
        auto auto_device_desc = get_description(device_args.addressable_devices[i]);
        std::string tmp = auto_device_desc;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](auto c){return std::tolower(c);});
        if (tmp.find("cpu") != std::string::npos) {
            chosen_device_idx = i;
            break;
        }
    }

    return device_args.addressable_devices[chosen_device_idx];
}

void execute_kernel(
    const PJRT_Api* api, 
    PJRT_LoadedExecutable* exe,
    PJRT_Client* client,
    PJRT_Device* device,
    const float* a_ptr,
    const float* b_ptr,
    float* o_ptr,
    long vector_size   
) {
    auto getFromHostBuffer = [&](const float* ptr) -> PJRT_Buffer* {
        PJRT_Client_BufferFromHostBuffer_Args buffer_args = {};
        buffer_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        buffer_args.client = client;
        buffer_args.data = ptr;
        int64_t dims_arr[] = {vector_size}; // TODO: ?
        buffer_args.dims = dims_arr;
        buffer_args.num_dims = 1;
        auto err = api -> PJRT_Client_BufferFromHostBuffer(&buffer_args);
        if (err) {
            std::cerr << "[ERR] faill to create host side buffer: " << get_err_msg(api, err) << std::endl;
            return nullptr;
        } 
        return buffer_args.buffer;
    };

    // TODO: using event can make this part async
    auto saveBackToHostBuffer = [&](float* ptr) -> void {
        PJRT_Buffer_ToHostBuffer_Args buffer_args = {};
        buffer_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        buffer_args.dst = ptr;
        buffer_args.dst_size = vector_size * 4; // TODO: currently I hardcode the size consider its f32
        auto err = api->PJRT_Buffer_ToHostBuffer(&buffer_args);
        if (err) {
            std::cout << "[ERR] fail to save back to the to host buffer: " << get_err_msg(api, err) << std::endl; 
            return;
        }
        return;
    };


    PJRT_LoadedExecutable_Execute_Args leeas = {};
    leeas.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    // function and args 
    leeas.executable = exe;
    PJRT_ExecuteOptions execute_options = {};
    execute_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    leeas.options = &execute_options;

    leeas.num_devices = (size_t) 1;
    leeas.num_args = (size_t) 3;
    PJRT_Buffer* input_buffers[] = {getFromHostBuffer(a_ptr), getFromHostBuffer(b_ptr)};
    PJRT_Buffer* const* device_input_list[] = {input_buffers};
    leeas.argument_lists = device_input_list;
    // we have one device, and the output by this device is 1.
    PJRT_Buffer** device_0_output = (PJRT_Buffer**)malloc(1 * sizeof(PJRT_Buffer*)); // we have one output
    PJRT_Buffer*** output_lists = (PJRT_Buffer***)malloc(1 * sizeof(PJRT_Buffer**)); // we have one device
    output_lists[0] = device_0_output;
    leeas.output_lists = output_lists;
    leeas.execute_device = device;  

    PJRT_Error* error;
    std::cout << "checkpoint: to be deleted" << std::endl;
    error = api->PJRT_LoadedExecutable_Execute(&leeas);
    if (error) {
        std::cerr << "[ERR] fail to execute " << get_err_msg(api, error) << std::endl;
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

    // TODO: delete output_lists PJRT_Buffer....!!!!
}


void check_null(void* ptr) {
    if (!ptr) {
       exit(1); 
    }
    return;
}

// TODO: argument number and vector shape (dimension) are hardcoded here, need to change later
void ExecuteMLIR(
    std::string func_code,
    const float* a_ptr,
    const float* b_ptr,
    float* o_ptr,
    long vector_size    
) {
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

    auto client = create_client(api);
    check_null(client);

    auto cpu_device = findCPUDevice(api, client);
    check_null(client);

    auto exe = compile_mlir(api, client, func_code);
    check_null(exe);

    execute_kernel(api, exe, client, cpu_device, a_ptr, b_ptr, o_ptr, vector_size);

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
    ExecuteMLIR(op, a_float_ptr, b_float_ptr, o_float_ptr, n);
    /**
        End: Using real addition through XLA
     */
}
