#include "runtime/jit-manager.h"

#include "../support/profiler.h"
#include "../support/utilities.h"
#include "jit-manager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

/**
-------------------- Tool Functions --------------------
 */

// FIXME: used for debugging only, delete this after usage
static void printBufferShape(const PJRT_Api *api, PJRT_Buffer *buffer,
                             const char *label, int device, int arg) {
  PJRT_Buffer_Dimensions_Args q{};
  q.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  q.buffer = buffer;
  if (auto *err = api->PJRT_Buffer_Dimensions(&q)) {
    llvm::errs() << JitManager::getErrMsg(api, err) << "\n";
    return;
  }
  llvm::errs() << label << " device=" << device << " arg=" << arg << " [";
  for (size_t d = 0; d < q.num_dims; ++d) {
    llvm::errs() << (d ? "," : "") << q.dims[d];
    llvm::errs() << "]\n";
  }
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
      .dims = inputArg.shape.data(),
      .num_dims = size_t(inputArg.rank),
      .element_type = getPJRTBufferType(inputArg.dtype),
      // PJRT_Buffer_MemoryLayout* layout;
      // TODO: use memory instead of device
      .device = device,
      .on_delete_callback = doNothingCallback,
  };
  auto *err = api->PJRT_Client_CreateViewOfDeviceBuffer(&cvodbArg);
  if (err) {
    std::cerr << "[Error] Fail to create View Buffer: "
              << JitManager::getInstance().getErrMsg(api, err) << "\n";
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
      .dims = inputArg.shape.data(),
      .num_dims = size_t(inputArg.rank),
      .host_buffer_semantics = PJRT_HostBufferSemantics_kMutableZeroCopy,
      .device = device};
  auto *err = api->PJRT_Client_BufferFromHostBuffer(&args);
  if (err) {
    std::cerr << "[Error] Fail to create CPU Buffer: "
              << JitManager::getInstance().getErrMsg(api, err) << "\n";
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
  DTypeVal val = extractLiteralPtr(rawPtr, dataType);

  void *host_ptr = nullptr;
  switch (val.returnedType) {
  case DType::I32:
    host_ptr = &val.valI32;
    break;
  case DType::I64:
    host_ptr = &val.valI64;
    break;
  case DType::F32:
    host_ptr = &val.valF32;
    break;
  case DType::F64:
    host_ptr = &val.valF64;
    break;
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

static PJRT_Buffer *createBufferFromForgedTgtPointers(
    const PJRT_Api *api, PJRT_Client *client, int deviceIdx, int deviceCount,
    PJRT_Device *device, const TensorDesc &inputArg,
    uint32_t offsetInByte = 0) {
  // pointing to a memory on host, host does not know the size
  // To make it work on TPU, we have to do it here
  void *dataSrc = inputArg.data;
  auto &internalBufferMap = getInternalBufferMap();
  auto it = internalBufferMap.find(dataSrc);
  if (it == internalBufferMap.end()) {
    internalBufferMap[dataSrc] =
        std::vector<PJRT_Buffer *>(deviceCount, nullptr);
  }
  assert(internalBufferMap.find(dataSrc) != internalBufferMap.end());
  std::vector<PJRT_Buffer *> &buffers = internalBufferMap[dataSrc];
  assert(buffers.size() > deviceIdx);
  assert(buffers.size() == deviceCount);

  if (!buffers[deviceIdx]) {
    auto args = PJRT_Client_BufferFromHostBuffer_Args{
        .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
        .client = client,
        .data = static_cast<void *>(static_cast<std::byte *>(dataSrc) +
                                    offsetInByte),
        .type = getPJRTBufferType(inputArg.dtype),
        .dims = inputArg.shape.data(),
        .num_dims = size_t(inputArg.rank),
        .device = device};
    auto err = api->PJRT_Client_BufferFromHostBuffer(&args);
    assert(!err);
    buffers[deviceIdx] = args.buffer;
  }

  return buffers[deviceIdx];
}

// For filtering out the target device.
static std::string getDeviceDescription(const PJRT_Api *api,
                                        PJRT_Device *device) {
  PJRT_Device_GetDescription_Args args = {
      .struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE,
      .device = device,
  };
  auto err1 = api->PJRT_Device_GetDescription(&args);
  if (err1) {
    std::cerr << "[Error] Fail to get description of the device: "
              << JitManager::getErrMsg(api, err1) << "\n";
    return nullptr;
  }
  PJRT_DeviceDescription_ToString_Args ts_args = {
      .struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE,
      .device_description = args.device_description,
  };
  auto err2 = api->PJRT_DeviceDescription_ToString(&ts_args);
  if (err2) {
    std::cerr << "[Error] Fail to get device description to string: "
              << JitManager::getErrMsg(api, err2) << "\n";
    return nullptr;
  }
  return ts_args.to_string;
}

// Get the target device handle
static llvm::SmallVector<PJRT_Device *>
findDevices(const PJRT_Api *api, PJRT_Client *client,
            const std::string &deviceDescKeyword) {
  PJRT_Client_AddressableDevices_Args device_args = {
      .struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
      .client = client,
  };
  auto err = api->PJRT_Client_AddressableDevices(&device_args);
  if (err) {
    std::cerr << "[Error] Cannot get addressable device: "
              << JitManager::getInstance().getErrMsg(api, err) << "\n";
    std::exit(EXIT_FAILURE);
  }
  if (device_args.num_addressable_devices < 1) {
    std::cerr << "[Error] Cannot find any device!\n";
    std::exit(EXIT_FAILURE);
  }

  llvm::SmallVector<PJRT_Device *> matched_devices;
  for (int i = 0; i < device_args.num_addressable_devices; i++) {
    auto device = device_args.addressable_devices[i];
    std::string tmp = getDeviceDescription(api, device);
    DEBUG_PRINT("Read device description: " + tmp +
                ", and trying to find device: " + deviceDescKeyword);
    std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    DEBUG_PRINT("After lower: " + tmp);
    if (tmp.find(deviceDescKeyword) != std::string::npos) {
      matched_devices.push_back(device);
    }
  }
  DEBUG_PRINT(
      llvm::formatv("Found {0} matched PJRT devices.", matched_devices.size()));
  return matched_devices;
}


DeviceManager::DeviceManager(const PJRT_Api* api, PJRT_Client* client, TargetDeviceType targetDeviceTy): 
  api_(api), client_(client), targetDeviceTy_(targetDeviceTy) {
  // Find all the available devices in the environment.
  switch (this->targetDeviceTy_) {
  case TargetDeviceType::CPU:
    DEBUG_PRINT("Trying to get device type: CPU.");
    this->pjrtDevices_ = findDevices(api_, client_, "cpu");
    break;
  case TargetDeviceType::CUDA:
    DEBUG_PRINT("Trying to get device type: CUDA.");
    this->pjrtDevices_ = findDevices(api_, client_, "cuda");
    break;
  case TargetDeviceType::ROCM:
    DEBUG_PRINT("Trying to get device type: ROCM.");
    this->pjrtDevices_ = findDevices(api_, client_, "rocm");
    break;
  case TargetDeviceType::TPU:
    DEBUG_PRINT("Trying to get device type: TPU.");
    this->pjrtDevices_ = findDevices(api_, client_, "tpu");
    break;
  default:
    DEBUG_PRINT("Trying to get unknown type device, device type is: " +
                std::to_string((int32_t)this->targetDeviceTy_));
    std::cerr << "Fail to find device!\n";
    std::exit(EXIT_FAILURE);
  }
};


// // Creating PJRT Buffers for inputs.
// // For normal tensor input value, creating view buffer to achieve zero copy.
// // For literal value, we have to create buffer and move.
// // For CPU, we need special setting to avoid XLA do extra copy, see
// // `createCPUBuffer`. For TPU, buffer is already created.
// static void manageInputBuffers(const PJRT_Api *api, PJRT_Client *client,
//                                int deviceIdx, int deviceCount,
//                                PJRT_Device *device, TargetDeviceType
//                                deviceType, const TensorDesc *inputArgs, const
//                                int32_t inputArgCount, std::vector<PJRT_Buffer
//                                *> &buffers) {
//   // PROFILE_SCOPE("manageInputBuffers", Phase::EXECUTION_BUFFER_PREPARE);
//   assert(buffers.size() == inputArgCount &&
//          "Buffer size should be the same with arg counts");
//   if (deviceType == TargetDeviceType::CPU) {
//     for (int i = 0; i < inputArgCount; i++) {
//       if (inputArgs[i].isLiteral) {
//         buffers[i] = createLiteralBuffer(api, client, device, inputArgs[i]);
//       } else {
//         buffers[i] = createCPUBuffer(api, client, device, inputArgs[i]);
//       }
//     }
//   } else if (deviceType == TargetDeviceType::TPU) {
//     for (int i = 0; i < inputArgCount; i++) {
//       if (inputArgs[i].isLiteral) {
//         buffers[i] = createLiteralBuffer(api, client, device, inputArgs[i]);
//       } else {
//         buffers[i] = createBufferFromForgedTgtPointers(
//             api, client, deviceIdx, deviceCount, device, inputArgs[i]);
//       }
//     }
//   } else {
//     for (int i = 0; i < inputArgCount; i++) {
//       if (inputArgs[i].isLiteral) {
//         buffers[i] = createLiteralBuffer(api, client, device, inputArgs[i]);
//       } else {
//         buffers[i] = createViewBuffer(api, client, device, inputArgs[i]);
//       }
//     }
//   }
//   return;
// }
//
// static void executeLoadedKernelExecutable(
//     const PJRT_Api *api, PJRT_LoadedExecutable *exe, PJRT_Device *device,
//     PJRT_Buffer ***argLists, PJRT_Buffer ***outLists, const int
//     in_args_count) {
//   PROFILE_SCOPE("executeExecutable", Phase::EXECUTION_RUN);
//
//   PJRT_ExecuteOptions execute_options = {
//       .struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
//   };
//
//   const int deviceCount = 1;
//
//   PJRT_Event *deviceCompleteEvents[deviceCount] = {nullptr};
//
//   PJRT_LoadedExecutable_Execute_Args leeas = {
//       .struct_size =
//           PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE, // function and
//           args
//       .executable = exe,
//       .options = &execute_options,
//       .argument_lists = argLists,         // [deviceCount][argCount],
//       .num_devices = (size_t)deviceCount, // we have one device, and the
//       output
//                                           // by this device is 1.
//       .num_args = (size_t)in_args_count,
//       .output_lists = outLists,
//       .device_complete_events = deviceCompleteEvents,
//       .execute_device = device,
//   };
//
//   auto executeErr = api->PJRT_LoadedExecutable_Execute(&leeas);
//   if (executeErr) {
//     std::cerr << "[Error] Execute LoadedExecutable: "
//               << JitManager::getErrMsg(api, executeErr) << "\n";
//     std::exit(EXIT_FAILURE);
//   }
//
//   for (int i = 0; i < deviceCount; i++) {
//     if (leeas.device_complete_events != nullptr &&
//         leeas.device_complete_events[i] != nullptr) {
//       PJRT_Event_Await_Args waitArgs = {
//           .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
//           .event = leeas.device_complete_events[i]};
//       api->PJRT_Event_Await(&waitArgs);
//       PJRT_Event_Destroy_Args eda = {.struct_size =
//                                          PJRT_Event_Destroy_Args_STRUCT_SIZE,
//                                      .event =
//                                      leeas.device_complete_events[i]};
//       api->PJRT_Event_Destroy(&eda);
//     }
//   }
// }
//
// /// Ideally, the data should be updated in-place
// /// But if not, we need to copy the data
// static void manageOutputBuffers(const PJRT_Api *api,
//                                 const std::vector<PJRT_Buffer *>
//                                 &argsBuffers, PJRT_Buffer ***outsBuffersList,
//                                 TensorDesc *inputArgs, TensorDesc
//                                 *outputArgs, TargetDeviceType targetDeviceTy)
//                                 {
//   // PROFILE_SCOPE("manageOutputBuffers", Phase::EXECUTION_BUFFER_CLEARUP);
//   for (int i = 0; i < argsBuffers.size(); i++) {
//     auto inputArg = inputArgs[i];
//     if (inputArg.isLiteral) {
//       destroyPJRTBuffer(api, outsBuffersList[0][i]);
//       continue;
//     }
//
//     PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args odmdpArgs = {
//         .struct_size =
//             PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE,
//         .buffer = outsBuffersList[0][i],
//     };
//     api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&odmdpArgs);
//     void *afterPtr = odmdpArgs.device_memory_ptr;
//     if (afterPtr != inputArg.data) {
//       if (targetDeviceTy == TargetDeviceType::CPU) {
//         DEBUG_PRINT("Data copied! arg idx: " + std::to_string(i));
//         // llvm::dbgs() << "Data copied to " << outputArgs[i].data << "\n";
//         std::memcpy(outputArgs[i].data, afterPtr,
//                     inputArg.getEleSize() *
//                     getDTypeSizeInByte(inputArg.dtype));
//       } else if (targetDeviceTy == TargetDeviceType::CUDA) {
//         // TODO: insert cudamemd2d?
//         std::cerr << "Not Implemented Yet for CUDA!\n";
//         exit(EXIT_FAILURE);
//       } else if (targetDeviceTy == TargetDeviceType::TPU) {
//         // This is supposed to be happen, because the inputArg.data is not a
//         // real pointer on the device, but a pointer we forged on OpenMP side
//         to
//         // the buffer.
//         auto &InternalBufferMap = getInternalBufferMap();
//         InternalBufferMap[inputArg.data][0] = outsBuffersList[0][i];
//         continue;
//       } else {
//         std::cerr << "[Error] Unsupported Device: "
//                   << std::to_string(static_cast<int32_t>(targetDeviceTy))
//                   << "\n";
//         exit(EXIT_FAILURE);
//       }
//       destroyPJRTBuffer(api, outsBuffersList[0][i]);
//     }
//   }
// }
//
// // TODO: currently, launching the whole kernel in a single device.
// void JitManager::launchKernel(PJRT_LoadedExecutable *exe,
//                               KernelArgs *offloadingArgs,
//                               const std::string &kernelFuncStr) {
//   assert(!this->pjrtDevices.empty());
//   auto device = this->pjrtDevices[0];
//
//   auto deviceType = JitManager::getInstance().targetDeviceTy;
//   // Create Buffer with memory managed by OpenMP
//   std::vector<PJRT_Buffer *> inputArgsBufs;
//   inputArgsBufs.resize(offloadingArgs->inputArgCount);
//   manageInputBuffers(pjrtApi, pjrtClient, 0, pjrtDevices.size(), device,
//   deviceType,
//                      offloadingArgs->inputArgs,
//                      offloadingArgs->inputArgCount, inputArgsBufs);
//   PJRT_Buffer **inputArgsBufsList[] = {inputArgsBufs.data()};
//
//   std::vector<PJRT_Buffer *> outputArgsBufs;
//   outputArgsBufs.resize(offloadingArgs->outputArgCount);
//   PJRT_Buffer **outputArgsBufsList[] = {outputArgsBufs.data()};
//
//   // Execute the kernel
//   executeLoadedKernelExecutable(this->pjrtApi, exe, device,
//   inputArgsBufsList,
//                                 outputArgsBufsList,
//                                 offloadingArgs->inputArgCount);
//   manageOutputBuffers(this->pjrtApi, inputArgsBufs, outputArgsBufsList,
//                       offloadingArgs->inputArgs, offloadingArgs->outputArgs,
//                       deviceType);
//   return;
// }

/**
 * following functions are used for executing in multiple devices
 * Should merge the logic with the functions above after testing!
 */

static void manageMultiDevicesInputBuffers(
    const PJRT_Api *api, PJRT_Client *client,
    llvm::SmallVector<PJRT_Device *> devices, TargetDeviceType deviceType,
    const TensorDesc *inputArgs, const int32_t inputArgCount,
    std::vector<std::vector<PJRT_Buffer *>> &buffers, int partitionCount,
    int replicaCount) {
  // PROFILE_SCOPE("manageInputBuffers", Phase::EXECUTION_BUFFER_PREPARE);
  assert(buffers.size() == devices.size() &&
         "Buffer size should be the same with devices size");
  assert(buffers.size() > 0);
  assert(buffers[0].size() == inputArgCount &&
         "Buffer size of any device should be the same with arg counts");
  if (deviceType == TargetDeviceType::CPU) {
    llvm::errs() << "Do not use CPU for this test, as PJRT device is different "
                    "from physical device. Multiple device execution on CPU "
                    "does not make a lot of sense.\n";
    // but do not panic
  }
  for (int devIdx = 0; devIdx < devices.size(); devIdx++) {
    auto device = devices[devIdx];
    for (int i = 0; i < inputArgCount; i++) {
      if (inputArgs[i].isLiteral) {
        buffers[devIdx][i] =
            createLiteralBuffer(api, client, device, inputArgs[i]);
      } else {
        if (replicaCount == 1 && partitionCount > 1) {
          TensorDesc slicedArg = inputArgs[i]; // sliceArg supposed to be a copy
          auto offset = devIdx * slicedArg.getDTypeSize() *
                        (slicedArg.getEleCount() / partitionCount);
          slicedArg.shape[0] = slicedArg.shape[0] / partitionCount;
          buffers[devIdx][i] = createBufferFromForgedTgtPointers(
              api, client, devIdx, devices.size(), device, slicedArg, offset);
        } else {
          buffers[devIdx][i] = createBufferFromForgedTgtPointers(
              api, client, devIdx, devices.size(), device, inputArgs[i]);
        }
      }

      // FIXME: delete after debugging
      printBufferShape(api, buffers[devIdx][i], "[DEBUG IN]", devIdx, i);
    }
  }
  return;
}

static void executeLoadedKernelExecutableOnMultiDevices(
    const PJRT_Api *api, PJRT_LoadedExecutable *exe,
    llvm::SmallVector<PJRT_Device *> devices, PJRT_Buffer ***argLists,
    PJRT_Buffer **const *outLists, const int in_args_count) {
  PROFILE_SCOPE("executeExecutable", Phase::EXECUTION_RUN);
  PJRT_ExecuteOptions execute_options = {
      .struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
  };
  const int deviceCount = devices.size();
  // int deviceCount = 2;
  std::vector<PJRT_Event *> deviceCompleteEvents(deviceCount);

  PJRT_LoadedExecutable_Execute_Args leeas = {
      .struct_size =
          PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE, // function and args
      .executable = exe,
      .options = &execute_options,
      .argument_lists = argLists, // [deviceCount][argCount],
      .num_devices = (size_t)deviceCount,
      .num_args = (size_t)in_args_count, // WARNING: what does this mean?
      .output_lists = outLists,
      .device_complete_events = deviceCompleteEvents.data(),
      .execute_device = nullptr,
  };

  auto executeErr = api->PJRT_LoadedExecutable_Execute(&leeas);
  if (executeErr) {
    std::cerr << "[Error] Execute LoadedExecutable: "
              << JitManager::getErrMsg(api, executeErr) << "\n";
    std::exit(EXIT_FAILURE);
  }

  for (int i = 0; i < deviceCount; i++) {
    if (leeas.device_complete_events != nullptr &&
        leeas.device_complete_events[i] != nullptr) {
      PJRT_Event_Await_Args waitArgs = {
          .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
          .event = leeas.device_complete_events[i]};
      api->PJRT_Event_Await(&waitArgs);
      PJRT_Event_Destroy_Args eda = {.struct_size =
                                         PJRT_Event_Destroy_Args_STRUCT_SIZE,
                                     .event = leeas.device_complete_events[i]};
      api->PJRT_Event_Destroy(&eda);
    }
  }
}

static void manageMultiDevicesOutputBuffers(
    const PJRT_Api *api, int inArgsCount, PJRT_Buffer **const *outsBuffersList,
    TensorDesc *inputArgs, TensorDesc *outputArgs,
    TargetDeviceType targetDeviceTy, uint32_t deviceCount) {
  auto &internalBufferMap = getInternalBufferMap();
  for (int devIdx = 0; devIdx < deviceCount; devIdx++) {
    for (int i = 0; i < inArgsCount; i++) {
      auto inputArg = inputArgs[i];
      if (inputArg.isLiteral) {
        destroyPJRTBuffer(api, outsBuffersList[devIdx][i]);
        continue;
      }

      PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args odmdpArgs = {
          .struct_size =
              PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE,
          .buffer = outsBuffersList[devIdx][i],
      };
      api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&odmdpArgs);
      void *afterPtr = odmdpArgs.device_memory_ptr;

      internalBufferMap[inputArg.data][devIdx] = outsBuffersList[devIdx][i];

      // FIXME: delete after debugging
      printBufferShape(api, outsBuffersList[devIdx][i], "[DEBUG OUT]", devIdx,
                       i);
    }
  }
}

void DeviceManager::launchKernelOnMultiDevices(PJRT_LoadedExecutable *exe,
                                            KernelArgs *offloadingArgs,
                                            const std::string &kernelFuncStr) {

  DEBUG_PRINT("Enter launchKernelOnMultiDevices");
  auto devices = this->pjrtDevices_;

  // InputBufs[deviceId][argIdx]
  std::vector<std::vector<PJRT_Buffer *>> inputBufs(
      devices.size(),
      std::vector<PJRT_Buffer *>(offloadingArgs->inputArgCount));

  // Move data from input data `offloading -> inputArgs` to `inputBufs`.
  // InputBufs are buffers in each devices.
  DEBUG_PRINT("Before manage input buffers");
  manageMultiDevicesInputBuffers(
      api_, client_, devices, targetDeviceTy_,
      offloadingArgs->inputArgs, offloadingArgs->inputArgCount, inputBufs,
      PARTITION_COUNT, REPLICA_COUNT);
  DEBUG_PRINT("Finish manage input buffers");

  PJRT_Buffer ***inputBufsList;
  std::vector<PJRT_Buffer **> rawInputBufs(inputBufs.size());
  for (int i = 0; i < inputBufs.size(); i++) {
    rawInputBufs[i] = inputBufs[i].data();
  }
  inputBufsList = rawInputBufs.data();

  std::vector<std::vector<PJRT_Buffer *>> outputArgsBufs(
      devices.size(),
      std::vector<PJRT_Buffer *>(offloadingArgs->outputArgCount));

  // outputBufsList stores the data handler for each output on each device.
  // [device][argIdx]
  PJRT_Buffer **const *outputBufsList;
  std::vector<PJRT_Buffer **> rawOutputBufs(outputArgsBufs.size());
  for (int i = 0; i < outputArgsBufs.size(); i++) {
    rawOutputBufs[i] = outputArgsBufs[i].data();
  }
  outputBufsList = rawOutputBufs.data();

  DEBUG_PRINT("Before executing");
  executeLoadedKernelExecutableOnMultiDevices(api_, exe, devices,
                                              inputBufsList, outputBufsList,
                                              offloadingArgs->inputArgCount);
  DEBUG_PRINT("Finish executing");

  // move data back to the host
  manageMultiDevicesOutputBuffers(api_, offloadingArgs->inputArgCount,
                                  outputBufsList, offloadingArgs->inputArgs,
                                  offloadingArgs->outputArgs, targetDeviceTy_,
                                  devices.size());
  DEBUG_PRINT("Finish manage out buffers");
  return;
}



void DeviceManager::moveDataToHostBuffer(void *hostPtr, size_t size) {
  auto bufferMap = getInternalBufferMap();
  auto it = bufferMap.find(hostPtr);
  if (it == bufferMap.end()) {
    llvm::errs() << "Can not find related buffer!\n";
    return;
  }
  const auto& buffers = it->second;
  size_t offset = 0;
  // TODO: delete hard coded part after test
  size_t bufferSize = size/buffers.size();

  llvm::SmallVector<PJRT_Event*> events(buffers.size());
  for (int devIdx = 0; devIdx < buffers.size(); devIdx++) {
    PJRT_Buffer* buffer = buffers.at(devIdx);

    auto args = PJRT_Buffer_ToHostBuffer_Args {
      .struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
      .src = buffer,
      .dst = static_cast<void*>(static_cast<std::byte*>(hostPtr) + offset),
      .dst_size = bufferSize
    };
    auto* err = api_->PJRT_Buffer_ToHostBuffer(&args);
    assert(!err);
    events[devIdx] = args.event;
    offset += bufferSize;
  }

  for (int devIdx = 0; devIdx < buffers.size(); devIdx++) {
    auto awaitArgs = PJRT_Event_Await_Args{
      .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
      .event = events[devIdx] 
    };
    auto* err2 = api_->PJRT_Event_Await(&awaitArgs);
    assert(!err2);
  }
}

void DeviceManager::destroyHostBoundBuffers(void* hostPtr) {
  auto &InternalBufferMap = getInternalBufferMap();
  auto it = InternalBufferMap.find(hostPtr);
  if (it != InternalBufferMap.end()) {
    for (auto bufferPtr : it->second) {
      PJRT_Buffer_Destroy_Args args = {.struct_size =
                                           PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
                                       .extension_start = nullptr,
                                       .buffer = bufferPtr};
      api_->PJRT_Buffer_Destroy(&args);
    }
    InternalBufferMap.erase(it);
  }
};
