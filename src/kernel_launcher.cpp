#include "jit-manager.h"
#include "kernel_pointer_interface.h"
#include "llvm/Support/Debug.h"
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
#include <vector>

/**
-------------------- Tool Functions --------------------
 */
std::string JitManager::getErrMsg(const PJRT_Api *api, PJRT_Error *err) {
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

bool JitManager::checkPJRTError(const PJRT_Api *api, PJRT_Error *err,
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
  if (!JitManager::checkPJRTError(api, api->PJRT_Client_CreateViewOfDeviceBuffer(&cvodbArg), "Create View of Device Buffer")) {
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
  auto raw = reinterpret_cast<uintptr_t>(inputArg.data);
  auto b = JitManager::getInstance().getLiteralBuffer(device, raw, inputArg.dtype);
  if (!b) {
    std::cerr << "Fail to get buffer for literal ptr!\n";
    std::exit(EXIT_FAILURE);
  }
  return b;
}

static void manageInputBuffers(
  const PJRT_Api *api,
  PJRT_Client* client,
  PJRT_Device* device,
  const TensorDesc* inputArgs,
  const int32_t inputArgCount,
  std::vector<PJRT_Buffer*>& buffers
) {
  assert(buffers.size() == inputArgCount && "Buffer size should be the same with arg counts");
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
  JitManager::checkPJRTError(api, executeErr, "Execute LoadedExecutable");

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

/// Ideally, the data should be updated in-place
/// But if not, we need to copy the data
static void manageOutputBuffers(
  const PJRT_Api* api, 
  const std::vector<PJRT_Buffer*>& argsBuffers,
  PJRT_Buffer*** argsBuffersList,
  TensorDesc* inputArgs,
  TensorDesc* outputArgs,
  TargetDevice targetDeviceTy
) {
  for (int i = 0; i < argsBuffers.size(); i++) {
    auto inputArg = inputArgs[i];
    // We don't care about literal arg, as we'll always get from cache or create them 
    if (inputArg.isLiteral) {
      return;
    }
    
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args odmdpArgs = {
      .struct_size = PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE,
      .buffer = argsBuffersList[0][i],
    };
    api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&odmdpArgs);
    void* afterPtr = odmdpArgs.device_memory_ptr;
    if (afterPtr != inputArg.data) {
      if (targetDeviceTy == TargetDevice::CPU) {
        llvm::dbgs() << "Data copied to " << outputArgs[i].data << "\n";
        std::memcpy(outputArgs[i].data, afterPtr, inputArg.getEleSize() * getDTypeSizeInByte(inputArg.dtype));
      } else if (targetDeviceTy == TargetDevice::CUDA) {
        // TODO: insert cudamemd2d 
        std::cerr << "Not Implemented Yet for CUDA!\n";
        exit(EXIT_FAILURE);   
      } else {
        logger::Log("Unsupported Device: " + std::to_string(static_cast<int32_t>(targetDeviceTy)), logLevel::ERROR);
        exit(EXIT_FAILURE);
      }
    }
  }
}

void JitManager::launchKernel(
  PJRT_LoadedExecutable* exe,
  KernelArgs *offloadingArgs, 
  const uintptr_t JitCodePtr, 
  const std::string &kernelFuncStr
) {
  auto device = this->getPJRTDevice(offloadingArgs->targetDevice);
  
  // Create Buffer with memory managed by OpenMP
  std::vector<PJRT_Buffer*> argsBuffers;
  argsBuffers.resize(offloadingArgs->inputArgCount);

  manageInputBuffers(this->pjrtApi, this->pjrtClient, device, offloadingArgs->inputArgs, offloadingArgs->inputArgCount, argsBuffers);
  PJRT_Buffer** argsBuffersList[] = {argsBuffers.data()};
  // Execute the kernel
  executeLoadedKernelExecutable(this->pjrtApi, exe, device, argsBuffersList, argsBuffersList, offloadingArgs->inputArgCount);
  
  manageOutputBuffers(this->pjrtApi, argsBuffers, argsBuffersList, offloadingArgs->inputArgs, offloadingArgs->outputArgs, offloadingArgs->targetDevice);
  return;
}


/// ---------------------------------------------------------------------

[[deprecated("Memory Allocation is done by OpenMP RT")]]
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
  if (!JitManager::checkPJRTError(api, err, "Create Buffer From Host")) {
    return nullptr;
  }
  return buffer_args.buffer;
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
  if (!JitManager::checkPJRTError(api, err, "Save buffer to host")){
    return;
  }
  if (buffer_args.event!= nullptr) {
    PJRT_Event_Await_Args await_args = {};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.event = buffer_args.event;
    JitManager::checkPJRTError(api, api->PJRT_Event_Await(&await_args), "Waiting for host buffer copy");

    PJRT_Event_Destroy_Args destroy_args = {};
    destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_args.event = buffer_args.event;
    api->PJRT_Event_Destroy(&destroy_args);
  }
}

