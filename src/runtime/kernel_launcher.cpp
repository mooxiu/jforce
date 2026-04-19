#include "jit-manager.h"
#include "../support/kernel_pointer_interface.h"
#include "../support/profiler.h"
#include <iostream>

static std::unordered_map<void *, PJRT_Buffer *> InternalBufferMap;

extern "C" {
__attribute__((visibility("default"))) PJRT_Buffer *
GetPjrtBuffer(void *cpu_ptr) {
  auto it = InternalBufferMap.find(cpu_ptr);
  if (it != InternalBufferMap.end()) {
    return it->second;
  }
  return nullptr;
}

__attribute__((visibility("default"))) void DestroyPjrtBuffer(void *cpu_ptr,
                                                              PJRT_Api *api) {
  auto it = InternalBufferMap.find(cpu_ptr);
  if (it != InternalBufferMap.end()) {
    PJRT_Buffer_Destroy_Args args = {PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
                                     nullptr, it->second};
    api->PJRT_Buffer_Destroy(&args);
    InternalBufferMap.erase(it);
  }
}
}

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

/**
-------------------- End Tool Functions --------------------
 */
static bool destroyPJRTBuffer(const PJRT_Api *api, PJRT_Buffer *dataPtr) {
  PJRT_Buffer_Destroy_Args args = {
      .struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE, .buffer = dataPtr};

  api->PJRT_Buffer_Destroy(&args);
  return true;
}

static PJRT_Buffer *createViewBuffer(const PJRT_Api *api, PJRT_Client *client,
                                     PJRT_Device *device,
                                     const TensorDesc &inputArg) {

  auto doNothingCallback = [](void *a, void *b) {};
  PJRT_Client_CreateViewOfDeviceBuffer_Args cvodbArg = {
    .struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE,
    .client = client,
    .device_buffer_ptr = inputArg.data,
    .dims = inputArg.shape,
    .num_dims = size_t(inputArg.rank),
    .element_type = getPJRTBufferType(inputArg.dtype),
    // PJRT_Buffer_MemoryLayout* layout;
    // TODO: use memory instead of device
    .device = device,
    .on_delete_callback = doNothingCallback,
  };
  auto* err = api->PJRT_Client_CreateViewOfDeviceBuffer(&cvodbArg);
  if (err) {
    std::cerr << "[Error] Fail to create View Buffer: " << JitManager::getInstance().getErrMsg(api, err) << "\n";
    std::exit(EXIT_FAILURE);
  }
  return cvodbArg.buffer;
}

static PJRT_Buffer *createCPUBuffer(const PJRT_Api *api, PJRT_Client *client,
                                    PJRT_Device *device,
                                    const TensorDesc &inputArg) {
  PJRT_Client_BufferFromHostBuffer_Args args = {
      .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
      .client = client,
      .data = inputArg.data,
      .type = getPJRTBufferType(inputArg.dtype),
      .dims = inputArg.shape,
      .num_dims = size_t(inputArg.rank),
      .host_buffer_semantics = PJRT_HostBufferSemantics_kMutableZeroCopy,
      .device = device};
  auto* err = api->PJRT_Client_BufferFromHostBuffer(&args);
  if (err) {
    std::cerr << "[Error] Fail to create CPU Buffer: " << JitManager::getInstance().getErrMsg(api, err) << "\n";
    std::exit(EXIT_FAILURE);
  }
  return args.buffer;
}

static PJRT_Buffer *createLiteralBuffer(const PJRT_Api *api,
                                        PJRT_Client *client,
                                        PJRT_Device *device,
                                        const TensorDesc &inputArg) {
  uintptr_t rawPtr = reinterpret_cast<uintptr_t>(inputArg.data);
  DType dataType = inputArg.dtype;

  int32_t val_i32;
  int64_t val_i64;
  float val_f32;
  double val_f64;

  void *host_ptr = nullptr;

  switch (dataType) {
  case DType::I32: {
    val_i32 = static_cast<int32_t>(rawPtr);
    host_ptr = &val_i32;
    break;
  }
  case DType::I64: {
    val_i64 = static_cast<int64_t>(rawPtr);
    host_ptr = &val_i64;
    break;
  }
  case DType::F32: {
    uint32_t low_bits = static_cast<uint32_t>(rawPtr);
    std::memcpy(&val_f32, &low_bits, sizeof(float));
    host_ptr = &val_f32;
    break;
  }
  case DType::F64: {
    std::memcpy(&val_f64, &rawPtr, sizeof(double));
    host_ptr = &val_f64;
    break;
  }
  }

  PJRT_Client_BufferFromHostBuffer_Args buffer_args = {};
  buffer_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  buffer_args.type = getPJRTBufferType(dataType);

  int64_t dims[1] = {};
  buffer_args.client = client;
  buffer_args.data = host_ptr;
  buffer_args.dims = dims;
  buffer_args.num_dims = 0; // TODO: should reconsider how to set the size and
                            // dimmension for general
  buffer_args.device = device;

  auto err = api->PJRT_Client_BufferFromHostBuffer(&buffer_args);
  if (err) {
    std::cerr << "Fail to create literal buffer from host!\n";
    return nullptr;
  }

  return buffer_args.buffer;
}

static PJRT_Buffer *
createBufferFromForgedTgtPointers(const PJRT_Api *api, PJRT_Client *client,
                                  PJRT_Device *device,
                                  const TensorDesc &inputArg) {
  // pointing to a memory on host, host does not know the size
  // To make it work on TPU, we have to do it here
  auto forgedPointer = inputArg.data;
  auto it = InternalBufferMap.find(forgedPointer);
  if (it == InternalBufferMap.end()) {
    int64_t dims_arr[inputArg.rank];
    for (int i = 0; i < inputArg.rank; i++) {
      dims_arr[i] = inputArg.shape[i];
    }

    auto args = PJRT_Client_BufferFromHostBuffer_Args{
        .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
        .client = client,
        .data = inputArg.data,
        .type = getPJRTBufferType(inputArg.dtype),
        .dims = dims_arr,
        .num_dims = size_t(inputArg.rank),
        .device = device};
    auto err = api->PJRT_Client_BufferFromHostBuffer(&args);
    assert(!err);
    InternalBufferMap[inputArg.data] = args.buffer;
    return args.buffer;
  } else {
    return it->second;
  }
}

static void manageInputBuffers(const PJRT_Api *api, PJRT_Client *client,
                               PJRT_Device *device, TargetDevice targetDevice,
                               const TensorDesc *inputArgs,
                               const int32_t inputArgCount,
                               std::vector<PJRT_Buffer *> &buffers) {
  PROFILE_SCOPE("manageInputBuffers", Phase::EXECUTION_BUFFER_PREPARE);
  assert(buffers.size() == inputArgCount &&
         "Buffer size should be the same with arg counts");
  if (targetDevice == TargetDevice::CPU) {
    for (int i = 0; i < inputArgCount; i++) {
      if (inputArgs[i].isLiteral) {
        buffers[i] = createLiteralBuffer(api, client, device, inputArgs[i]);
      } else {
        buffers[i] = createCPUBuffer(api, client, device, inputArgs[i]);
      }
    }
  } else if (targetDevice == TargetDevice::TPU) {
    // In TPU, buffer is already created by the offload plugin!
    // We should not create View Buffer, but instead, we should reuse the
    // buffer.
    for (int i = 0; i < inputArgCount; i++) {
      if (inputArgs[i].isLiteral) {
        buffers[i] = createLiteralBuffer(api, client, device, inputArgs[i]);
      } else {
        buffers[i] = createBufferFromForgedTgtPointers(api, client, device,
                                                       inputArgs[i]);
      }
    }
  } else {
    for (int i = 0; i < inputArgCount; i++) {
      if (inputArgs[i].isLiteral) {
        buffers[i] = createLiteralBuffer(api, client, device, inputArgs[i]);
      } else {
        buffers[i] = createViewBuffer(api, client, device, inputArgs[i]);
      }
    }
  }
  return;
}

static void executeLoadedKernelExecutable(
    const PJRT_Api *api, PJRT_LoadedExecutable *exe, PJRT_Device *device,
    PJRT_Buffer ***argLists, PJRT_Buffer ***outLists, const int in_args_count) {
  PROFILE_SCOPE("executeExecutable", Phase::EXECUTION_RUN);

  PJRT_ExecuteOptions execute_options = {
      .struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
  };

  const int deviceCount = 1;

  PJRT_Event *deviceCompleteEvents[deviceCount] = {nullptr};

  PJRT_LoadedExecutable_Execute_Args leeas = {
      .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE, // function and args
      .executable = exe,
      .options = &execute_options,
      .argument_lists = argLists,         // [deviceCount][argCount],
      .num_devices = (size_t)deviceCount, // we have one device, and the output by this device is 1.
      .num_args = (size_t)in_args_count,
      .output_lists = outLists,
      .device_complete_events = deviceCompleteEvents,
      .execute_device = device,
  };

  auto executeErr = api->PJRT_LoadedExecutable_Execute(&leeas);
  if (executeErr) {
    std::cerr << "[Error] Execute LoadedExecutable: " 
      << JitManager::getErrMsg(api, executeErr) << "\n";
    std::exit(EXIT_FAILURE);
  }

  for (int i = 0; i < deviceCount; i++) {
    if (leeas.device_complete_events != nullptr && leeas.device_complete_events[i] != nullptr) {
      PJRT_Event_Await_Args waitArgs = {
        .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
        .event = leeas.device_complete_events[i]
      };
      api->PJRT_Event_Await(&waitArgs);
      PJRT_Event_Destroy_Args eda = {
        .struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE,
        .event = leeas.device_complete_events[i]
      };
      api->PJRT_Event_Destroy(&eda);
    }
  }
}

/// Ideally, the data should be updated in-place
/// But if not, we need to copy the data
static void manageOutputBuffers(const PJRT_Api *api,
                                const std::vector<PJRT_Buffer *> &argsBuffers,
                                PJRT_Buffer ***outsBuffersList,
                                TensorDesc *inputArgs, TensorDesc *outputArgs,
                                TargetDevice targetDeviceTy) {
  PROFILE_SCOPE("manageOutputBuffers", Phase::EXECUTION_BUFFER_CLEARUP);
  for (int i = 0; i < argsBuffers.size(); i++) {
    auto inputArg = inputArgs[i];
    if (inputArg.isLiteral) {
      destroyPJRTBuffer(api, outsBuffersList[0][i]);
      continue;
    }

    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args odmdpArgs = {
        .struct_size =
            PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE,
        .buffer = outsBuffersList[0][i],
    };
    api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&odmdpArgs);
    void *afterPtr = odmdpArgs.device_memory_ptr;
    if (afterPtr != inputArg.data) {
      if (targetDeviceTy == TargetDevice::CPU) {
        DEBUG_PRINT("Data copied! arg idx: " + std::to_string(i));
        // llvm::dbgs() << "Data copied to " << outputArgs[i].data << "\n";
        std::memcpy(outputArgs[i].data, afterPtr,
                    inputArg.getEleSize() * getDTypeSizeInByte(inputArg.dtype));
      } else if (targetDeviceTy == TargetDevice::CUDA) {
        // TODO: insert cudamemd2d?
        std::cerr << "Not Implemented Yet for CUDA!\n";
        exit(EXIT_FAILURE);
      } else if (targetDeviceTy == TargetDevice::TPU) {
        // This is supposed to be happen, because the inputArg.data is not a real pointer on the device, 
        // but a pointer we forged on OpenMP side to the buffer.
        extern std::unordered_map<void *, PJRT_Buffer *> InternalBufferMap;
        InternalBufferMap[inputArg.data] = outsBuffersList[0][i];
        continue;
      } else {
        std::cerr << "[Error] Unsupported Device: " << std::to_string(static_cast<int32_t>(targetDeviceTy)) << "\n";
        exit(EXIT_FAILURE);
      }
      destroyPJRTBuffer(api, outsBuffersList[0][i]);
    }
  }
}

void JitManager::launchKernel(PJRT_LoadedExecutable *exe,
                              KernelArgs *offloadingArgs,
                              const uintptr_t JitCodePtr,
                              const std::string &kernelFuncStr) {
  auto device = this->getPJRTDevice(offloadingArgs->targetDevice);

  // Create Buffer with memory managed by OpenMP
  std::vector<PJRT_Buffer *> inputArgsBufs;
  inputArgsBufs.resize(offloadingArgs->inputArgCount);
  manageInputBuffers(this->pjrtApi, this->pjrtClient, device,
                     offloadingArgs->targetDevice, offloadingArgs->inputArgs,
                     offloadingArgs->inputArgCount, inputArgsBufs);
  PJRT_Buffer **inputArgsBufsList[] = {inputArgsBufs.data()};

  std::vector<PJRT_Buffer *> outputArgsBufs;
  outputArgsBufs.resize(offloadingArgs->outputArgCount);
  PJRT_Buffer **outputArgsBufsList[] = {outputArgsBufs.data()};

  // Execute the kernel
  executeLoadedKernelExecutable(this->pjrtApi, exe, device, inputArgsBufsList,
                                outputArgsBufsList,
                                offloadingArgs->inputArgCount);
  manageOutputBuffers(this->pjrtApi, inputArgsBufs, outputArgsBufsList,
                      offloadingArgs->inputArgs, offloadingArgs->outputArgs,
                      offloadingArgs->targetDevice);
  return;
}

/// ---------------------------------------------------------------------
//
// [[deprecated("Memory Allocation is done by OpenMP RT")]]
// static size_t getSizeOf(PJRT_Buffer_Type type) {
//   switch (type) {
//   case PJRT_Buffer_Type_F32:
//   case PJRT_Buffer_Type_S32:
//     return size_t(4);
//   case PJRT_Buffer_Type_F64:
//   case PJRT_Buffer_Type_S64:
//     return size_t(8);
//   default:
//     logger::Log("Unknown Type", logLevel::ERROR);
//     exit(1);
//   }
// }
//
// // TODO: This work should later be done by OpenMP runtime.
// [[deprecated("Handled by OpenMP runtime, don't need to assign the buffer by "
//              "ourseleves")]]
// static PJRT_Buffer *getBufferFromHost(const PJRT_Api *api, PJRT_Client *client,
//                                       PJRT_Device *device, void *ptr,
//                                       std::vector<int64_t> shape) {
//   PJRT_Client_BufferFromHostBuffer_Args buffer_args = {
//       .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
//       .client = client,
//       .data = ptr,
//       .type = PJRT_Buffer_Type_F32,
//       .num_dims = shape.size(), // Should reconsider how to set the size and
//                                 // dimmension for general
//       .device = device,
//   };
//
//   int64_t dims_arr[shape.size()];
//   for (int i = 0; i < shape.size(); i++) {
//     dims_arr[i] = shape[i];
//   }
//   buffer_args.dims = dims_arr;
//   auto err = api->PJRT_Client_BufferFromHostBuffer(&buffer_args);
//   if (!JitManager::checkPJRTError(api, err, "Create Buffer From Host")) {
//     return nullptr;
//   }
//   return buffer_args.buffer;
// }
//
// // TODO: need to fix because OpenMP will manage the memory location of the host
// // and device
// [[deprecated("OpenMP will in charge of the memory")]]
// static void saveBufferToHostBuffer(const PJRT_Api *api, PJRT_Buffer *source,
//                                    void *dst, std::vector<int64_t> shape) {
//   PJRT_Buffer_ToHostBuffer_Args buffer_args = {};
//   buffer_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
//   buffer_args.src = source;
//   buffer_args.dst = dst;
//   buffer_args.dst_size = getSizeOf(PJRT_Buffer_Type_F32) *
//                          std::accumulate(shape.begin(), shape.end(), 1,
//                                          std::multiplies<int64_t>());
//   buffer_args.event = nullptr;
//
//   auto err = api->PJRT_Buffer_ToHostBuffer(&buffer_args);
//   if (!JitManager::checkPJRTError(api, err, "Save buffer to host")) {
//     return;
//   }
//   if (buffer_args.event != nullptr) {
//     PJRT_Event_Await_Args await_args = {};
//     await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
//     await_args.event = buffer_args.event;
//     JitManager::checkPJRTError(api, api->PJRT_Event_Await(&await_args),
//                                "Waiting for host buffer copy");
//
//     PJRT_Event_Destroy_Args destroy_args = {};
//     destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
//     destroy_args.event = buffer_args.event;
//     api->PJRT_Event_Destroy(&destroy_args);
//   }
// }
