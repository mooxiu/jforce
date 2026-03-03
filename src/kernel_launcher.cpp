#include "kernel_launcher.h"
#include "llvm/Support/Debug.h"
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
#include <numeric>
#include <ostream>
#include <string>

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
    return true;
  }
}

/**
-------------------- End Tool Functions --------------------
 */


static PJRT_Client *getPJRTClient(const PJRT_Api *api) {
  if (client) {
    return client;
  } else {
    PJRT_Client_Create_Args args = {};
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    auto error = api->PJRT_Client_Create(&args);
    if (!checkPJRTError(api, error, "Creating Client")) {
      return nullptr;
    }
    client = args.client;
    return client;
  }
}

[[deprecated("Client has lifetime through the whole program, it will be automatically destroyed")]]
static void destroyClient(const PJRT_Api *api, PJRT_Client *client) {
  PJRT_Client_Destroy_Args client_destroy_args = {};
  client_destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  client_destroy_args.client = client;
  api->PJRT_Client_Destroy(&client_destroy_args);
  return;
}

[[deprecated("Lifetime through the whole program")]]
static void destroyLoadedExecutable(const PJRT_Api *api, PJRT_LoadedExecutable *exe) {
  PJRT_LoadedExecutable_Destroy_Args ledargs = {
    .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
    .executable = exe,
  };
  auto destroyErr = api->PJRT_LoadedExecutable_Destroy(&ledargs);
  checkPJRTError(api, destroyErr, "Destroy LoadedExecutable");
  return;
}

// For filtering out the target device.
static std::string getDeviceDescription(const PJRT_Api *api, PJRT_Device *device) {
  PJRT_Device_GetDescription_Args args = {
    .struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE,
    .device = device,
  };
  auto err1 = api->PJRT_Device_GetDescription(&args);
  if (err1) {
    logger::Log("Fail to get description of device: " + getErrMsg(api, err1),
                logLevel::ERROR);
    return nullptr;
  }
  PJRT_DeviceDescription_ToString_Args ts_args = {
    .struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE,
    .device_description = args.device_description,
  };
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
  PJRT_Client_AddressableDevices_Args device_args = {
    .struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
    .client = client,
  };
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
  return device_args.addressable_devices[chosen_device_idx];
}

static PJRT_Device* getPJRTDevice(const PJRT_Api* api, PJRT_Client *client, KernelArgs *offloadingArgs) {
  if (device) {
    return device;
  } else {
    if (offloadingArgs->targetDevice == TargetDevice::CPU) {
      device = findDevice(api, client, "cpu");
    } else if (offloadingArgs->targetDevice == TargetDevice::CUDA){
      device = findDevice(api, client, "cuda");
    } else {
      std::cerr << "Unsupported Device Type!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!device) {
      std::cerr << "Device not find! \n";
      std::exit(EXIT_FAILURE);
    }
    return device;
  }
}

static void destroyPJRTBuffer(const PJRT_Api *api, PJRT_Buffer *buffer) {
  PJRT_Buffer_Destroy_Args args = {
    .struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
    .buffer = buffer
  };
  auto err = api->PJRT_Buffer_Destroy(&args);
  checkPJRTError(api, err, "Destroy Buffer");
  return;
}

// TODO: This work should later be done by OpenMP runtime.
[[deprecated("Handled by OpenMP runtime, don't need to assign the buffer by ourseleves")]]
static PJRT_Buffer *getBufferFromHost(const PJRT_Api *api, PJRT_Client *client,
                               PJRT_Device *device, void *ptr,
                               std::vector<int64_t> shape) {
  PJRT_Client_BufferFromHostBuffer_Args buffer_args = {
    .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
    .client = client,
    .data = ptr,
    .type = PJRT_Buffer_Type_F32,
    .num_dims = shape.size(), // Should reconsider how to set the size and dimmension for general
    .device = device,
  };
  
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
    case PJRT_Buffer_Type_S32:
      return size_t(4);
    case PJRT_Buffer_Type_F64:
    case PJRT_Buffer_Type_S64:
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
  const TensorDesc& inputArg
) {

  PJRT_Client_CreateViewOfDeviceBuffer_Args cvodbArg= {};
  cvodbArg.client = client;
  cvodbArg.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
  cvodbArg.element_type = [&](){
    switch (inputArg.dtype) {
      case DType::F32:
        return PJRT_Buffer_Type_F32;
      case DType::F64:
        return PJRT_Buffer_Type_F64;
      case DType::I32:
        return PJRT_Buffer_Type_S32;
      case DType::I64:
        return PJRT_Buffer_Type_S64;
      default:
        std::cerr << "[Error] Unexecpted data type: " << std::to_string((int32_t)inputArg.dtype) << "\n";
        exit(EXIT_FAILURE);
    }
  }();
  // TODO: use memory instead of device
  cvodbArg.device = device;
  cvodbArg.device_buffer_ptr = inputArg.data;
  cvodbArg.num_dims= inputArg.rank;
  cvodbArg.dims = inputArg.shape;
  auto doNothingCallback = [](void* a, void* b){};
  cvodbArg.on_delete_callback = doNothingCallback;
  if (!checkPJRTError(api, api->PJRT_Client_CreateViewOfDeviceBuffer(&cvodbArg), "Create View of Device Buffer")) {
    std::cerr << "Fail to create View of Device Buffer! Exit...\n";  
    std::exit(EXIT_FAILURE);
  };
  return cvodbArg.buffer;
}

// Creating Buffer for single number is not recommended, but preprocessing (folding) it in stableHLO!
static PJRT_Buffer *createLiteralBuffers(
  const PJRT_Api *api, 
  PJRT_Client *client,
  PJRT_Device *device, 
  const TensorDesc& inputArg
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
      case DType::I64: {
        int64_t* dataptrI64 = (int64_t*)malloc(sizeof(int64_t) * 1);
        int64_t val = static_cast<int64_t>(raw);
        *dataptrI64 = val;
        return (void*)dataptrI64; 
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
    } else if (inputArg.dtype == DType::I64) {
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S64;
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

static void executeLoadedKernelExecutable(
  const PJRT_Api *api, 
  PJRT_LoadedExecutable *exe,
  PJRT_Device *device, 
  PJRT_Buffer ***argLists,
  PJRT_Buffer ***outLists,
  const int in_args_count
) {
  PJRT_ExecuteOptions execute_options = {
    .struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
  };

  const int deviceCount = 1;

  PJRT_Event* deviceCompleteEvents[deviceCount];

  PJRT_LoadedExecutable_Execute_Args leeas = {
    .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,// function and args
    .executable = exe,
    .options = &execute_options,
    .argument_lists = argLists, // [deviceCount][argCount], 
    .num_devices = (size_t)deviceCount, // we have one device, and the output by this device is 1.
    .num_args = (size_t)in_args_count,
    .output_lists = outLists,
    .device_complete_events = deviceCompleteEvents,
    .execute_device = device,
  };
  
  auto executeErr = api->PJRT_LoadedExecutable_Execute(&leeas);
  checkPJRTError(api, executeErr, "Execute LoadedExecutable");

  for(int i = 0; i < deviceCount; i++) {
    PJRT_Event_Await_Args waitArgs = {
      .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
      .event = leeas.device_complete_events[i]
    };
    api->PJRT_Event_Await(&waitArgs);
  } 

  for (int i = 0; i < deviceCount; i++) {
    PJRT_Event_Destroy_Args eda = {
      .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
      .event = leeas.device_complete_events[i]
    };
    api->PJRT_Event_Destroy(&eda);
  }
}

static void cpyMemOnDevice(void* dstPtr, void* srcPrt, size_t byteCount, TargetDevice deviceTy) {
  if (deviceTy == TargetDevice::CPU) {
    std::memcpy(dstPtr, srcPrt, byteCount);
  } else if (deviceTy == TargetDevice::CUDA) {
    // TODO: insert cudamemd2d 
    llvm::dbgs() << "Device Allocated Ptr: " << dstPtr << ", Function OutputPtr: " << srcPrt << ". Size: " << byteCount << "\n";
    std::cerr << "Not Implemented Yet!\n";
    exit(EXIT_FAILURE);
  } else {
    logger::Log("Unsupported Device: " + std::to_string(static_cast<int32_t>(deviceTy)), logLevel::ERROR);
    exit(EXIT_FAILURE);
  }
}

void launchKernel(
  KernelArgs *offloadingArgs, 
  const uintptr_t JitCodePtr, 
  const std::string &kernelFuncStr
) {
  const PJRT_Api* api = JitManager::getInstance().getPJRTApi();
  auto client = getPJRTClient(api);
  auto device = getPJRTDevice(api, client, offloadingArgs);
  
  PJRT_LoadedExecutable* exe = JitManager::getInstance().getPJRTExecutable(api, client, kernelFuncStr, offloadingArgs, JitCodePtr);

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
  executeLoadedKernelExecutable(api, exe, device, argsBuffersList, argsBuffersList, offloadingArgs->inputArgCount);


  // Already wait in above execution!
  // PJRT_Buffer_ReadyEvent_Args* eventArgs[offloadingArgs->inputArgCount];
  // for (int i = 0; i < offloadingArgs->inputArgCount; i++) {
  //   PJRT_Buffer_ReadyEvent_Args eventArg = {};
  //   eventArgs[i] = &eventArg;
  //   eventArg.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
  //   eventArg.buffer = argsBuffersList[0][i];
  //   api->PJRT_Buffer_ReadyEvent(&eventArg);
  // }
  //
  // for (int i = 0; i < offloadingArgs->inputArgCount; i++) {
  //   // Wait for the event to complete
  //   PJRT_Event_Await_Args waitArgs = {};
  //   waitArgs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  //   waitArgs.event = eventArgs[i]->event;
  //   api->PJRT_Event_Await(&waitArgs); 
  // }
  //

  // After Execution, the data may not be updated in-place!!!!!
  for (int i = 0; i < argsBuffers.size(); i++) {
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args odmdpArgs = {
      .struct_size = PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE,
      .buffer = argsBuffersList[0][i],
    };
    api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&odmdpArgs);
    void* afterPtr = odmdpArgs.device_memory_ptr;

    auto inputArg = offloadingArgs->inputArgs[i];
    if (!inputArg.isLiteral && afterPtr != inputArg.data) {
      cpyMemOnDevice(offloadingArgs->outputArgs[i].data, 
                  afterPtr, 
                  sizeof(float_t) * inputArg.getEleSize(),
                  offloadingArgs->targetDevice);
    }
  }

  for (int i = 0; i < argsBuffers.size(); i++) {
    PJRT_Buffer* b = argsBuffers.at(i);
    destroyPJRTBuffer(api, b);
  }
  return;
}

