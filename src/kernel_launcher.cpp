#include "../third_party/headers/pjrt_c_api.h"
#include "../third_party/protos/generated/xla/pjrt/proto/compile_options.pb.h"
#include "flang/Common/leading-zero-bit-count.h"
#include "kernel_pointer_interface.h"
#include "utilities.h"
#include "xla/xla.pb.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ostream>
#include <string>

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
static std::string getErrMsg(const PJRT_Api *api, PJRT_Error *err) {
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

static bool checkPJRTError(const PJRT_Api *api, PJRT_Error *err,
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

static PJRT_Api *getAPI() {
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

static PJRT_Client *createClient(const PJRT_Api *api) {
  PJRT_Client_Create_Args args = {};
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  auto error = api->PJRT_Client_Create(&args);
  if (!checkPJRTError(api, error, "Creating Client")) {
    return nullptr;
  }
  return args.client;
}

static void destroyClient(const PJRT_Api *api, PJRT_Client *client) {
  PJRT_Client_Destroy_Args client_destroy_args = {};
  client_destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  client_destroy_args.client = client;
  api->PJRT_Client_Destroy(&client_destroy_args);
  return;
}

static PJRT_LoadedExecutable *compileMLIR(const PJRT_Api *api, PJRT_Client *client,
                                   const std::string &func_code, KernelArgs* offloadingArgs) {
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

  auto getCompileOptionsProto = [&]() -> std::string {
    xla::CompileOptionsProto opts = {};
    opts.set_parameter_is_tupled_arguments(false);
    opts.set_compile_portable_executable(false);
    opts.set_profile_version(1);

    xla::ExecutableBuildOptionsProto *build_opts = opts.mutable_executable_build_options();
    build_opts->set_num_replicas(1);
    build_opts->set_num_partitions(1);

    // Special option for CUDA
    if (offloadingArgs->targetDevice == TargetDevice::CUDA) {
      // TODO: this might make compiled code slower!!!
      auto debugOptions = build_opts->mutable_debug_options();
      debugOptions->set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(true);
      if (const char* cuda_path_env = std::getenv("MY_CUDA_PATH")) {
        debugOptions->set_xla_gpu_cuda_data_dir(cuda_path_env);
        logger::Log("Setting cuda_data_dir to: " + std::string(cuda_path_env), logLevel::DEBUG);
      } else {
        // DO NOTHING, this might cause warning
      } 
    }    

    std::string buf;
    // SerializeToString(): This is protobuf's method inherited by `CompileOptionProto`.
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

static void destroyLoadedExecutable(const PJRT_Api *api, PJRT_LoadedExecutable *exe) {
  PJRT_LoadedExecutable_Destroy_Args ledargs;
  ledargs.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  ledargs.executable = exe;
  auto destroyErr = api->PJRT_LoadedExecutable_Destroy(&ledargs);
  checkPJRTError(api, destroyErr, "Destroy LoadedExecutable");
  return;
}

// For filtering out the target device.
static std::string getDeviceDescription(const PJRT_Api *api, PJRT_Device *device) {
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

// Get the target device handle
static PJRT_Device *findDevice(const PJRT_Api *api, PJRT_Client *client,
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

  // TODO: Delete this after testing
  // chosen_device_idx = 2;
  logger::Log("Have chosen device id: " + std::to_string(chosen_device_idx) +
                  " , desc: " + desc,
              logLevel::DEBUG);
  return device_args.addressable_devices[chosen_device_idx];
}

static void destroyPJRTBuffer(PJRT_Api *api, PJRT_Buffer *buffer) {
  PJRT_Buffer_Destroy_Args args = {};
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = buffer;
  auto err = api->PJRT_Buffer_Destroy(&args);
  checkPJRTError(api, err, "Destroy Buffer");
  return;
}

// TODO: This work should later be done by OpenMP runtime.
[[deprecated("Handled by OpenMP runtime, don't need to assign the buffer by ourseleves")]]
static PJRT_Buffer *getBufferFromHost(const PJRT_Api *api, PJRT_Client *client,
                               PJRT_Device *device, void *ptr,
                               std::vector<int64_t> shape) {
  PJRT_Client_BufferFromHostBuffer_Args buffer_args = {};
  buffer_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  buffer_args.type = PJRT_Buffer_Type_F32;
  buffer_args.device = device;
  buffer_args.client = client;
  buffer_args.data = ptr;
  // TODO: should reconsider how to set the size and dimmension for general
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
static size_t getSizeOf(PJRT_Buffer_Type type) {
  switch (type) {
  case PJRT_Buffer_Type_F32:
    return size_t(4);
  case PJRT_Buffer_Type_F64:
    return size_t(8);
  default:
    logger::Log("Unknown Type", logLevel::ERROR);
    exit(1);
  }
}

// TODO: need to fix because OpenMP will manage the memory location of the host and device
[[deprecated("OpenMP will in charge of the memory")]]
static void saveBufferToHostBuffer(const PJRT_Api *api, PJRT_Buffer *source, void *dst,
                            std::vector<int64_t> shape) {
  PJRT_Buffer_ToHostBuffer_Args buffer_args = {};
  buffer_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  buffer_args.src = source;
  buffer_args.dst = dst;
  buffer_args.dst_size = getSizeOf(PJRT_Buffer_Type_F32) *
                         std::accumulate(shape.begin(), shape.end(), 1,
                                         std::multiplies<int64_t>());
  buffer_args.event = nullptr;

  logger::Log("The size in byte is: " + std::to_string(buffer_args.dst_size),
              logLevel::DEBUG);
  auto err = api->PJRT_Buffer_ToHostBuffer(&buffer_args);
  if (!checkPJRTError(api, err, "Save buffer to host")){
    return;
  }
  if (buffer_args.event!= nullptr) {
    PJRT_Event_Await_Args await_args = {};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.event = buffer_args.event;
    checkPJRTError(api, api->PJRT_Event_Await(&await_args), "Waiting for host buffer copy");

    PJRT_Event_Destroy_Args destroy_args = {};
    destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_args.event = buffer_args.event;
    api->PJRT_Event_Destroy(&destroy_args);
  }
}

static PJRT_Buffer* createViewBuffers(
  const PJRT_Api *api,
  PJRT_Client* client,
  PJRT_Device* device,
  const TensorDesc inputArg
) {

  logger::Log("Start to create buffer", logLevel::DEBUG);
  PJRT_Client_CreateViewOfDeviceBuffer_Args cvodbArg= {};
  cvodbArg.client = client;
  cvodbArg.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
  cvodbArg.element_type = [&](){
    switch (inputArg.dtype) {
      case DType::F32:
        return PJRT_Buffer_Type_F32;
      case DType::F64:
        return PJRT_Buffer_Type_F64;
      default:
        logger::Log("Unexpected data type", logLevel::ERROR);
        exit(EXIT_FAILURE);
    }
  }();
  // TODO: use memory instead of device
  cvodbArg.device = device;
  cvodbArg.device_buffer_ptr = inputArg.data;
  cvodbArg.num_dims= inputArg.rank;
  cvodbArg.dims = inputArg.shape;
  auto doNothingCallback = [](void* a, void* b){
    std::cout << "Call onDeleteCallBack on ViewOfDeviceBuffer" << std::endl;
  };
  cvodbArg.on_delete_callback = doNothingCallback;
  checkPJRTError(api, api->PJRT_Client_CreateViewOfDeviceBuffer(&cvodbArg), "Create View of Device Buffer");
  return cvodbArg.buffer;
}

// Creating Buffer for single number is not recommended, but preprocessing (folding) it in stableHLO!
static PJRT_Buffer *createLiteralBuffers(
  const PJRT_Api *api, 
  PJRT_Client *client,
  PJRT_Device *device, 
  const TensorDesc inputArg
) {
  
  void* getLiteralData = [&]() -> void*{
    auto raw = reinterpret_cast<uintptr_t>(inputArg.data);
    switch (inputArg.dtype) {
      case DType::I32: {
        int32_t* dataptrI32 = (int32_t*)malloc(sizeof(int) * 1);
        int32_t val = static_cast<int32_t>(raw);
        *dataptrI32 = val;
        return (void*)dataptrI32; 
      }
      case DType::F64: {
        double_t* dataptrF64 = (double_t*)malloc(sizeof(double) * 1);
        memcpy(dataptrF64, &raw, sizeof(double));
        return (void*)dataptrF64;
      }
      case DType::F32: {
        uint32_t low_bits = static_cast<uint32_t>(raw);
        float_t* dataptrF32 = (float_t*)malloc(sizeof(float) * 1);
        memcpy(dataptrF32, &low_bits, sizeof(float));
        return (void*)dataptrF32;
      }
      default:
        return nullptr;
    }
  }();

  PJRT_Client_BufferFromHostBuffer_Args buffer_args = {};
  buffer_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  buffer_args.type = [&](){
    if (inputArg.dtype == DType::F32){
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F32;
    } else if (inputArg.dtype == DType::F64) {
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F64;
    } else if (inputArg.dtype == DType::I32) {
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S32;
    } else {
      std::cerr << "Unknown Buffer Types!\n";
      std::exit(EXIT_FAILURE);
    }
  }();
  buffer_args.device = device;
  buffer_args.client = client;
  buffer_args.data = getLiteralData;
  // TODO: should reconsider how to set the size and dimmension for general
  buffer_args.num_dims = 0;
  int64_t dims[1] = {};
  buffer_args.dims = dims;
  auto err = api->PJRT_Client_BufferFromHostBuffer(&buffer_args);
  if (!checkPJRTError(api, err, "Create Buffer From Host")) {
    return nullptr;
  }
  free(getLiteralData);
  return buffer_args.buffer;
}

static void manageInputBuffers(
  const PJRT_Api *api,
  PJRT_Client* client,
  PJRT_Device* device,
  const TensorDesc* inputArgs,
  const int32_t inputArgCount,
  std::vector<PJRT_Buffer*>& buffers,
  const std::vector<int32_t>& literalTypeArgsIndices) {

  buffers.resize(inputArgCount);
  for (int i = 0; i < inputArgCount; i++) {
    if (inputArgs[i].isLiteral){
      buffers[i] = createLiteralBuffers(api, client, device, inputArgs[i]);  
    } else {
      buffers[i] = createViewBuffers(api, client, device, inputArgs[i]);
    }
  }
  return;
}

static void executeKernel(
  const PJRT_Api *api, 
  PJRT_LoadedExecutable *exe,
  PJRT_Device *device, 
  PJRT_Buffer ***argLists,
  PJRT_Buffer ***outLists,
  const int in_args_count
) {
  PJRT_LoadedExecutable_Execute_Args leeas = {};
  leeas.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  // function and args
  leeas.executable = exe;
  PJRT_ExecuteOptions execute_options = {};
  execute_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;


  leeas.options = &execute_options;
  leeas.num_devices = (size_t)1;
  leeas.num_args = (size_t)in_args_count;
  leeas.argument_lists = argLists; // [deviceCount][argCount]
  // we have one device, and the output by this device is 1.
  leeas.output_lists = outLists;
  leeas.execute_device = device;

  auto executeErr = api->PJRT_LoadedExecutable_Execute(&leeas);
  checkPJRTError(api, executeErr, "Execute LoadedExecutable");
}

static void cpyMemOnDevice(void* dstPtr, void* srcPrt, size_t byteCount, TargetDevice deviceTy) {
  if (deviceTy == TargetDevice::CPU) {
    std::memcpy(dstPtr, srcPrt, byteCount);
  } else if (deviceTy == TargetDevice::CUDA) {
    // TODO: insert cudamemd2d 
  } else {
    logger::Log("Unsupported Device: " + std::to_string(static_cast<int32_t>(deviceTy)), logLevel::ERROR);
    exit(EXIT_FAILURE);
  }
}


static void launchKernelInternal(KernelArgs *offloadingArgs, const std::string& kernelFuncStr) {
  auto handle_ = dlopen(getPluginPath().c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  // auto handle_ = dlopen(getPluginPath().c_str(), RTLD_NOW|RTLD_GLOBAL);
  if (!handle_) {
    std::cerr << "error loading plugin: " << dlerror() << std::endl;
    return;
  }
  // follow the example of `man dlopen`
  auto get_api_fn = (PJRT_Api * (*)()) dlsym(handle_, "GetPjrtApi");
  if (!get_api_fn) {
    std::cerr << "error finding GetPjrtApi: " << dlerror() << std::endl;
    return;
  }
  auto api = get_api_fn();
  logger::Log("The API Loaded Successfully!", logLevel::DEBUG);

  auto checkNull = [handle_](void *ptr) -> void * {
    if (!ptr) {
      std::cerr << "[FATAL] Pointer not supposed to be null is null!! Exiting." << std::endl;
      exit(1);
    }
    return ptr;
  };

  PJRT_Plugin_Initialize_Args initArgs = {};
  initArgs.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  auto initErr = api->PJRT_Plugin_Initialize(&initArgs);
  checkPJRTError(api, initErr, "Init Plugins");

  auto client = (PJRT_Client *)checkNull(createClient(api));
  
  PJRT_Device* device;
  if (offloadingArgs->targetDevice == TargetDevice::CPU) {
    device = (PJRT_Device *)checkNull(findDevice(api, client, "cpu"));
  } else if (offloadingArgs->targetDevice == TargetDevice::CUDA){
    device = (PJRT_Device *)checkNull(findDevice(api, client, "cuda"));
  } else {
    std::cerr << "Unsupported Device Type!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Compile StableHLO to XLA kernel
  auto exe = (PJRT_LoadedExecutable *)checkNull(compileMLIR(api, client, kernelFuncStr, offloadingArgs));

  // Create Buffer with memory managed by OpenMP
  // For literal MapType, there's no memory been allocated, we have to allocate the memory and buffer by ourselves!!!! 
  std::vector<PJRT_Buffer*> argsBuffers;
  std::vector<int> literalTypeArgsIndices;
  literalTypeArgsIndices.reserve(offloadingArgs->inputArgCount);
  for (int32_t i = 0; i < offloadingArgs->inputArgCount; i++) {
    if (offloadingArgs->inputArgs[i].isLiteral) {
      literalTypeArgsIndices.push_back(i);
    }
  }
  manageInputBuffers(api, client, device, offloadingArgs->inputArgs, offloadingArgs->inputArgCount, argsBuffers, literalTypeArgsIndices);
  PJRT_Buffer** argsBuffersList[] = {argsBuffers.data()};


  // Execute the kernel
  executeKernel(api, exe, device, argsBuffersList, argsBuffersList, offloadingArgs->inputArgCount);

  PJRT_Buffer_ReadyEvent_Args* eventArgs[offloadingArgs->inputArgCount];
  for (int i = 0; i < offloadingArgs->inputArgCount; i++) {
    PJRT_Buffer_ReadyEvent_Args eventArg = {};
    eventArgs[i] = &eventArg;
    eventArg.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    eventArg.buffer = argsBuffersList[0][i];
    api->PJRT_Buffer_ReadyEvent(&eventArg);
  }
    
  for (int i = 0; i < offloadingArgs->inputArgCount; i++) {
    // Wait for the event to complete
    PJRT_Event_Await_Args waitArgs = {};
    waitArgs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    waitArgs.event = eventArgs[i]->event;
    api->PJRT_Event_Await(&waitArgs); 
  }


  // After Execution, the data may not be updated in-place!!!!!
  for (int i = 0; i < argsBuffers.size(); i++) {
    // PJRT_Buffer_UnsafePointer_Args upArgs = {};
    // upArgs.struct_size = PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE;
    // upArgs.buffer = argsBuffersList[0][i];
    // api->PJRT_Buffer_UnsafePointer(&upArgs);
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args odmdpArgs = {};
    odmdpArgs.struct_size = PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
    odmdpArgs.buffer = argsBuffersList[0][i];
    api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&odmdpArgs);

    void* afterPtr = odmdpArgs.device_memory_ptr;

    //--- print the comparsion of original pointers and new pointers
    std::cout << "Before Mem: " << offloadingArgs->inputArgs[i].data << std::endl;
    std::cout << "After Mem: " << afterPtr << std::endl;
    //---

    if (afterPtr != offloadingArgs->inputArgs[i].data) {
      logger::Log("Different Memory Address", logLevel::DEBUG);
      // TODO: what if this is in CUDA???????
      cpyMemOnDevice(offloadingArgs->outputArgs[i].data, 
                  afterPtr, 
                  sizeof(float_t) * offloadingArgs->inputArgs[i].getEleSize(),
                  offloadingArgs->targetDevice);
    }
  }


  // TODO: Still need to destroy PJRT_Buffers!!!!!!!
  destroyLoadedExecutable(api, exe);
  destroyClient(api, client);
  dlclose(handle_);

  return;
}

void launch_kernel(KernelArgs* kernelArgs, const std::string& kernelFuncStr) {
  launchKernelInternal(kernelArgs, kernelFuncStr);
}

