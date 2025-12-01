#include "../third_party/headers/pjrt_c_api.h"
#include "../third_party/protos/generated/xla/pjrt/proto/compile_options.pb.h"
#include "kernel_pointer_interface.h"
#include "operations.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

std::string getPluginPath() {
// DEFAULT_PJRT_PLUGIN_PATH should be defined in CMake
#ifdef DEFAULT_PJRT_PLUGIN_PATH
  return DEFAULT_PJRT_PLUGIN_PATH;
#else
  throw std::runtime_error(
      "PJRT plugin path not found. Please set PJRT_PLUGIN_PATH.");
#endif
}

/**
-------------------- Tool Functions --------------------
 */
std::string getErrMsg(const PJRT_Api *api, PJRT_Error *err) {
  PJRT_Error_GetCode_Args code_args = {};
  code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  code_args.error = err;

  api->PJRT_Error_GetCode(&code_args);

  PJRT_Error_Message_Args msg_args = {};
  msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  msg_args.error = err;
  api->PJRT_Error_Message(&msg_args);
  std::string s(msg_args.message);

  PJRT_Error_Destroy_Args destroy_args = {};
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.error = err;

  api->PJRT_Error_Destroy(&destroy_args);
  return s;
}

bool checkPJRTError(const PJRT_Api *api, PJRT_Error *err,
                    const std::string &eventName) {
  auto msg = eventName;
  if (err) {
    msg += " failed!";
    logger::Log(msg + ": " + getErrMsg(api, err), logLevel::ERROR);
    return false;
  } else {
    msg += " succeeded!";
    logger::Log(msg, logLevel::DEBUG);
    return true;
  }
}

/**
-------------------- End Tool Functions --------------------
 */

PJRT_Api *getAPI() {
  auto handle_ = dlopen(getPluginPath().c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle_) {
    logger::Log("Error loading plugin: " + std::string(dlerror()),
                logLevel::ERROR);
    return nullptr;
  }
  // follow the example of `man dlopen`
  auto get_api_fn = (PJRT_Api * (*)()) dlsym(handle_, "GetPjrtApi");
  if (!get_api_fn) {
    logger::Log("Error finding GetPjrtApi: " + std::string(dlerror()),
                logLevel::ERROR);
    return nullptr;
  }
  auto api = get_api_fn();
  logger::Log("The api loaded successfully!", logLevel::DEBUG);
  return api;
}

PJRT_Client *createClient(const PJRT_Api *api) {
  PJRT_Client_Create_Args args = {};
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  auto error = api->PJRT_Client_Create(&args);
  if (!checkPJRTError(api, error, "Creating Client")) {
    return nullptr;
  }
  return args.client;
}

void destroyClient(const PJRT_Api *api, PJRT_Client *client) {
  PJRT_Client_Destroy_Args client_destroy_args = {};
  client_destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  client_destroy_args.client = client;
  api->PJRT_Client_Destroy(&client_destroy_args);
  return;
}

PJRT_LoadedExecutable *compileMLIR(const PJRT_Api *api, PJRT_Client *client,
                                   const std::string &func_code) {
  auto setProgram = [func_code](PJRT_Program *program,
                                const std::string &format) -> void {
    program->struct_size = PJRT_Program_STRUCT_SIZE;
    // We have to set as mlir here as we're passing MLIR module string rather
    // than serialized HLOModuleProto.
    logger::Log("The code is \n" + func_code, logLevel::DEBUG);
    program->code = (char *)func_code.c_str();
    program->code_size = (size_t)func_code.size();
    program->format = format.c_str();
    program->format_size = (size_t)format.size();
    return;
  };

  auto getCompileOptionsProto = []() -> std::string {
    xla::CompileOptionsProto opts = {};
    opts.set_parameter_is_tupled_arguments(false);
    opts.set_compile_portable_executable(false);
    opts.set_profile_version(1);

    xla::ExecutableBuildOptionsProto *build_opts =
        opts.mutable_executable_build_options();
    build_opts->set_num_replicas(1);
    build_opts->set_num_partitions(1);

    std::string buf;
    // SerializeToString(): This is protobuf's method inherited by
    // CompileOptionProto.
    if (!opts.SerializeToString(&buf)) {
      logger::Log("Fail to serialize CompileOptionsProto", logLevel::ERROR);
      return nullptr;
    }
    return buf;
  };

  // It seems PJRT_Client_Compile will also help to load the execute
  PJRT_Client_Compile_Args compile_args = {};
  compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  compile_args.client = client;
  PJRT_Program program = {};
  setProgram(&program, "mlir");
  compile_args.program = &program;
  auto buf = getCompileOptionsProto();
  compile_args.compile_options = (char *)buf.c_str();
  compile_args.compile_options_size = (size_t)buf.size();

  auto error = api->PJRT_Client_Compile(&compile_args);
  if (!checkPJRTError(api, error, "Compile The Program")) {
    return nullptr;
  }
  return compile_args.executable;
}

// TODO: which one should I use? PJRT_LoadedExecutable_Delete or this?
void destroyLoadedExecutable(const PJRT_Api *api, PJRT_LoadedExecutable *exe) {
  PJRT_LoadedExecutable_Destroy_Args ledargs;
  ledargs.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  ledargs.executable = exe;
  auto destroyErr = api->PJRT_LoadedExecutable_Destroy(&ledargs);
  checkPJRTError(api, destroyErr, "Destroy LoadedExecutable");
  return;
}

std::string getDeviceDescription(const PJRT_Api *api, PJRT_Device *device) {
  PJRT_Device_GetDescription_Args args = {};
  args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
  args.device = device;
  auto err1 = api->PJRT_Device_GetDescription(&args);
  if (err1) {
    logger::Log("Fail to get description of device: " + getErrMsg(api, err1),
                logLevel::ERROR);
    return nullptr;
  }
  PJRT_DeviceDescription_ToString_Args ts_args = {};
  ts_args.struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
  ts_args.device_description = args.device_description;
  auto err2 = api->PJRT_DeviceDescription_ToString(&ts_args);
  if (err2) {
    logger::Log("Fail to get device description to string: " +
                    getErrMsg(api, err2),
                logLevel::ERROR);
    return nullptr;
  }
  return ts_args.to_string;
}

PJRT_Device *findDevice(const PJRT_Api *api, PJRT_Client *client,
                        const std::string &deviceDescKeyword) {
  PJRT_Client_AddressableDevices_Args device_args = {};
  device_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  device_args.client = client;
  auto err = api->PJRT_Client_AddressableDevices(&device_args);
  if (!checkPJRTError(api, err, "Find Device")) {
    return nullptr;
  }
  if (device_args.num_addressable_devices < 1) {
    logger::Log("Cannot find any device!", logLevel::ERROR);
    return nullptr;
  }

  int chosen_device_idx = -1;
  std::string desc = ""; // for logging purpose
  for (int i = 0; i < device_args.num_addressable_devices; i++) {
    auto auto_device_desc =
        getDeviceDescription(api, device_args.addressable_devices[i]);
    std::string tmp = auto_device_desc;
    std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                   [](auto c) { return std::tolower(c); });
    if (tmp.find(deviceDescKeyword) != std::string::npos) {
      chosen_device_idx = i;
      desc = tmp;
      break;
    }
  }
  if (chosen_device_idx == -1) {
    logger::Log("Fail to find " + deviceDescKeyword + " device!",
                logLevel::ERROR);
    return nullptr;
  }

  logger::Log("Have chosen device id: " + std::to_string(chosen_device_idx) +
                  " , desc: " + desc,
              logLevel::DEBUG);
  return device_args.addressable_devices[chosen_device_idx];
}

void destroyPJRTBuffer(PJRT_Api *api, PJRT_Buffer *buffer) {
  PJRT_Buffer_Destroy_Args args = {};
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = buffer;
  auto err = api->PJRT_Buffer_Destroy(&args);
  checkPJRTError(api, err, "Destroy Buffer");
  return;
}

// TODO: This work should later be done by OpenMP runtime.
PJRT_Buffer *getBufferFromHost(const PJRT_Api *api, PJRT_Client *client,
                               PJRT_Device *device, void *ptr,
                               std::vector<int64_t> shape) {
  PJRT_Client_BufferFromHostBuffer_Args buffer_args = {};
  buffer_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  buffer_args.type = PJRT_Buffer_Type_F32;
  buffer_args.device = device;
  buffer_args.client = client;
  buffer_args.data = ptr;
  // TODO: should reconsider how to set the size and dimmension for general
  // shape
  buffer_args.num_dims = shape.size();
  int64_t dims_arr[shape.size()];
  for (int i = 0; i < shape.size(); i++) {
    dims_arr[i] = shape[i];
  }
  buffer_args.dims = dims_arr;
  auto err = api->PJRT_Client_BufferFromHostBuffer(&buffer_args);
  if (!checkPJRTError(api, err, "Create Buffer From Host")) {
    return nullptr;
  }
  return buffer_args.buffer;
}

// TODO: should have a better implementation
size_t getSizeOf(PJRT_Buffer_Type type) {
  switch (type) {
  case PJRT_Buffer_Type_F32:
    return sizeof(float);
  default:
    logger::Log("Unknown Type", logLevel::ERROR);
    exit(1);
  }
}

// TODO: using event can make this part async
void saveBufferToHostBuffer(const PJRT_Api *api, PJRT_Buffer *source, void *dst,
                            std::vector<int64_t> shape) {
  PJRT_Buffer_ToHostBuffer_Args buffer_args = {};
  buffer_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  buffer_args.src = source;
  buffer_args.dst = dst;
  buffer_args.dst_size = getSizeOf(PJRT_Buffer_Type_F32) *
                         std::accumulate(shape.begin(), shape.end(), 1,
                                         std::multiplies<int64_t>());
  logger::Log("The size in byte is: " + std::to_string(buffer_args.dst_size),
              logLevel::DEBUG);
  auto err = api->PJRT_Buffer_ToHostBuffer(&buffer_args);
  checkPJRTError(api, err, "Save buffer to host");
  return;
}

void executeKernel(const PJRT_Api *api, PJRT_LoadedExecutable *exe,
                   PJRT_Device *device, PJRT_Buffer ***argLists,
                   PJRT_Buffer ***outLists) {
  PJRT_LoadedExecutable_Execute_Args leeas = {};
  leeas.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  // function and args
  leeas.executable = exe;
  PJRT_ExecuteOptions execute_options = {};
  execute_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  leeas.options = &execute_options;

  leeas.num_devices = (size_t)1;
  leeas.num_args = (size_t)2; // TODO: should accept general input args num

  leeas.argument_lists = argLists;

  // we have one device, and the output by this device is 1.
  leeas.output_lists = outLists;
  leeas.execute_device = device;

  auto executeErr = api->PJRT_LoadedExecutable_Execute(&leeas);
  checkPJRTError(api, executeErr, "Execute LoadedExecutable");
}

std::string getFuncCode(KernelArgs *args) {
  std::string funcCode = "";
  switch (args->opCode) {
  case OpType::VECTOR_ADD:
    funcCode = GetVectorAdditionOp(getShape(args->inputArgs[0])[0]);
    break;
  default:
    logger::Log("Unknown opCode: " + std::to_string((int32_t)args->opCode),
                logLevel::ERROR);
    exit(EXIT_FAILURE);
  }
  return funcCode;
}

void launchKernelInternal(KernelArgs *offloadingArgs) {
  auto handle_ = dlopen(getPluginPath().c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle_) {
    std::cerr << "error loading plugin: " << dlerror() << std::endl;
    return;
  }
  auto checkNull = [handle_](void *ptr) -> void * {
    if (!ptr) {
      // don't forget to clear the handle_ before panic
      dlclose(handle_);
      exit(1);
    }
    return ptr;
  };
  // follow the example of `man dlopen`
  auto get_api_fn = (PJRT_Api * (*)()) dlsym(handle_, "GetPjrtApi");
  if (!get_api_fn) {
    std::cerr << "error finding GetPjrtApi: " << dlerror() << std::endl;
    return;
  }
  auto api = get_api_fn();
  logger::Log("The API Loaded Successfully!", logLevel::DEBUG);

  auto client = (PJRT_Client *)checkNull(createClient(api));
  auto cpuDevice = (PJRT_Device *)checkNull(findDevice(api, client, "cpu"));

  auto exe = (PJRT_LoadedExecutable *)checkNull(
      compileMLIR(api, client, getFuncCode(offloadingArgs)));

  // Buffer from host
  int in_args_count = offloadingArgs->inputArgCount;
  PJRT_Buffer *inputArgsBuffers[in_args_count];
  for (int i = 0; i < in_args_count; i++) {
    inputArgsBuffers[i] = getBufferFromHost(
        api, client, cpuDevice, offloadingArgs->inputArgs[i].data,
        getShape(offloadingArgs->inputArgs[i]));
  }
  PJRT_Buffer **argLists[] = {inputArgsBuffers};

  // Set buffer save back to host
  int out_args_count = offloadingArgs->outputArgCount;
  PJRT_Buffer ***outputLists =
      (PJRT_Buffer ***)malloc(sizeof(PJRT_Buffer **)); // we have one device
  for (int i = 0; i < out_args_count; i++) {
    PJRT_Buffer **outBuffer =
        (PJRT_Buffer **)malloc(sizeof(PJRT_Buffer *)); // we have one output
    outputLists[i] = outBuffer;
  }

  // Execute the kernel
  executeKernel(api, exe, cpuDevice, argLists, outputLists);

  // TODO: actually should be able to save to multiple out
  for (int i = 0; i < out_args_count; i++) {
    saveBufferToHostBuffer(api, outputLists[0][i],
                           offloadingArgs->outputArgs[i].data,
                           getShape(offloadingArgs->outputArgs[i]));
  }

  // Destroy Input Events and Memory
  for (int i = 0; i < in_args_count; i++) {
    destroyPJRTBuffer(api, inputArgsBuffers[i]);
  }

  // Destroy Output Memory
  for (int i = 0; i < out_args_count; i++) {
    destroyPJRTBuffer(api, outputLists[0][i]);
  }

  for (int i = 0; i < out_args_count; i++) {
    free(outputLists[i]);
  }
  free(outputLists);

  destroyLoadedExecutable(api, exe);
  destroyClient(api, client);
  dlclose(handle_);

  return;
}

extern "C" void launch_kernel(void *argsPointer) {
  KernelArgs *kernelArgs = static_cast<KernelArgs *>(argsPointer);

  launchKernelInternal(kernelArgs);
}
