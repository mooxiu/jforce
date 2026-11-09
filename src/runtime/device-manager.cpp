#include "../support/utilities.h"
#include "jit-manager.h"
#include "runtime/jit-manager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <utility>
#include <vector>

static void destroyPJRTBuffer(const PJRT_Api *api, PJRT_Buffer *dataPtr) {
  PJRT_Buffer_Destroy_Args args = {
      .struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE, .buffer = dataPtr};
  api->PJRT_Buffer_Destroy(&args);
}

static void destroyPJRTBuffers(const PJRT_Api *api,
                               std::vector<DeviceBufferSlot> buffers) {
  for (auto &buffer : buffers) {
    destroyPJRTBuffer(api, buffer.bufferPtr);
  }
}

std::optional<std::vector<DeviceBufferSlot>>
DeviceBuffersCache::getBuffers(void *dataSrc) {
  std::shared_lock<std::shared_mutex> rLock(this->mtx);
  auto it = this->deviceBuffersMap.find(dataSrc);
  if (it != deviceBuffersMap.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<DeviceBufferSlot> DeviceBuffersCache::getBuffer(void *dataSrc,
                                                              int devIdx) {
  std::shared_lock<std::shared_mutex> rLock(this->mtx);
  auto it = this->deviceBuffersMap.find(dataSrc);
  if (it != deviceBuffersMap.end()) {
    auto buffers = it->second;
    if (buffers.size() - 1 < devIdx) {
      DEBUG_PRINT("DevIdx exceed the buffers size!\n");
      return std::nullopt;
    }
    return buffers.at(devIdx);
  }
  return std::nullopt;
}

// Try insert or update with buffer.
// If buffer already exists in the slot and the same with prepared buffer, do
// nothing. Else, insert or update.
DeviceBufferSlot DeviceBuffersCache::tryInsertOrUpdate(const PJRT_Api *api,
                                                       void *dataSrc,
                                                       int devIdx, int devCount,
                                                       DeviceBufferSlot slot) {
  if (devIdx > devCount - 1) {
    llvm::errs() << "Invalid devIdx and devCount.\n";
    std::exit(EXIT_FAILURE);
  }

  std::unique_lock<std::shared_mutex> wLock(this->mtx);
  auto it = deviceBuffersMap.find(dataSrc);
  if (it == deviceBuffersMap.end()) {
    // cache not exist
    it = deviceBuffersMap
             .emplace(dataSrc, std::vector<DeviceBufferSlot>(
                                   devCount, DeviceBufferSlot{nullptr}))
             .first;
  }
  auto &buffers = it->second;
  assert(buffers.size() == devCount);
  // The same buffer may come back with a new shard description: keep it.
  if (buffers[devIdx].bufferPtr != nullptr &&
      buffers[devIdx].bufferPtr != slot.bufferPtr) {
    destroyPJRTBuffer(api, buffers[devIdx].bufferPtr);
  }
  buffers[devIdx] = std::move(slot);
  return buffers[devIdx];
}

void DeviceBuffersCache::deleteEntry(void *dataSrc, const PJRT_Api *api_) {
  std::unique_lock<std::shared_mutex> wLock(this->mtx);
  auto it = deviceBuffersMap.find(dataSrc);
  if (it == deviceBuffersMap.end()) {
    return;
  }
  auto buffers = std::move(it->second);
  deviceBuffersMap.erase(it);
  wLock.unlock();
  destroyPJRTBuffers(api_, buffers);
}

bool DeviceManager::destroyPJRTBuffer(const PJRT_Api *api,
                                      PJRT_Buffer *dataPtr) {
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

PJRT_Buffer *DeviceManager::createLiteralBuffer(int devIdx,
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
  buffer_args.client = client_;
  buffer_args.data = host_ptr;
  buffer_args.dims = dims;
  buffer_args.num_dims = 0; // TODO: should reconsider how to set the size and
                            // dimmension for general
  buffer_args.device = pjrtDevices_[devIdx];

  auto err = api_->PJRT_Client_BufferFromHostBuffer(&buffer_args);
  if (err) {
    std::cerr << "Fail to create literal buffer from host!\n";
    return nullptr;
  }

  return buffer_args.buffer;
}

/// \param subtensorDims describes the shape of the selected part of the
/// original tensor to move. e.g., if we're moving T(1:3, 2:8, 3:7), then this
/// should be (2, 6, 4) = (3 - 1, 8 - 2, 7 - 3)
PJRT_Buffer *DeviceManager::createBufferFromForgedTgtPointers(
    int deviceIdx,
    // const TensorDesc &inputArg,
    void *dataSrc, DType elementDType, llvm::ArrayRef<int64_t> subtensorDims,
    llvm::ArrayRef<int64_t> byteStrides, uint32_t offsetInByte,
    llvm::ArrayRef<int64_t> fullShape, const ShardSlices &slices) {
  size_t rank = subtensorDims.size();
  // pointing to a memory on host, host does not know the size
  // To make it work on TPU, we have to do it here
  // FIXME(sharding-cache): the cached buffer is reused without checking its
  // sharding. It may be a previous kernel's output in a different sharding than
  // this kernel's input (per-kernel ILP, or outputs != inputs), so the local
  // shard may have the wrong shape or hold the wrong part of the tensor.
  auto cache = deviceBuffersCache.getBuffer(dataSrc, deviceIdx);
  if (cache.has_value() && cache.value().bufferPtr != nullptr) {
    return cache.value().bufferPtr;
  }
  auto args = PJRT_Client_BufferFromHostBuffer_Args{
      .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
      .client = client_,
      .data =
          static_cast<void *>(static_cast<std::byte *>(dataSrc) + offsetInByte),
      .type = getPJRTBufferType(elementDType),
      .dims = subtensorDims.data(),
      .num_dims = rank,
      .byte_strides = byteStrides.data(),
      .num_byte_strides = byteStrides.size(),
      .device = pjrtDevices_[deviceIdx]};
  auto err = api_->PJRT_Client_BufferFromHostBuffer(&args);
  if (err) {
    llvm::errs() << "BufferFromHostBuffer fails: "
                 << JitManager::getErrMsg(api_, err) << "\n";
    std::abort();
  }
  DeviceBufferSlot slot{args.buffer, slices,
                        llvm::SmallVector<int64_t>(fullShape),
                        getDTypeSizeInByte(elementDType)};
  auto updated = deviceBuffersCache.tryInsertOrUpdate(
      api_, dataSrc, deviceIdx, pjrtDevices_.size(), std::move(slot));
  return updated.bufferPtr;
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

DeviceManager::DeviceManager(const PJRT_Api *api, PJRT_Client *client,
                             TargetDeviceType targetDeviceTy)
    : api_(api), client_(client), targetDeviceTy_(targetDeviceTy) {
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

// /**
//  * following functions are used for executing in multiple devices
//  * Should merge the logic with the functions above after testing!
//  */
//
// void DeviceManager::manageMultiDevicesInputBuffers(
//     const TensorDesc *inputArgs, const int32_t inputArgCount,
//     std::vector<std::vector<PJRT_Buffer *>> &buffers,
//     const TensorShardings &sd) {
//   // PROFILE_SCOPE("manageInputBuffers", Phase::EXECUTION_BUFFER_PREPARE);
//   assert(buffers.size() == this->pjrtDevices_.size() &&
//          "Buffer size should be the same with devices size");
//   assert(buffers.size() > 0);
//   assert(buffers[0].size() == inputArgCount &&
//          "Buffer size of any device should be the same with arg counts");
//   if (this->targetDeviceTy_ == TargetDeviceType::CPU) {
//     llvm::errs() << "Do not use CPU for this test, as PJRT device is
//     different "
//                     "from physical device. Multiple device execution on CPU "
//                     "does not make a lot of sense.\n";
//     // but do not panic
//   }
//   for (int devIdx = 0; devIdx < this->pjrtDevices_.size(); devIdx++) {
//     auto device = this->pjrtDevices_[devIdx];
//     for (int i = 0; i < inputArgCount; i++) {
//       auto argSharding = sd[i];
//       if (argSharding.empty()) {
//         // argSharding is empty means this is a scale
//         buffers[devIdx][i] = createLiteralBuffer(api_, client_, device,
//         inputArgs[i]);
//       } else {
//         for (int i = 0; i < pjrtDevices_.size(); i++) {
//           buffers[devIdx][i] =
//           createBufferFromForgedTgtPointersWithSlice(devIdx, );
//         }
//       }
//     }
//   }
//   return;
// }
//

bool DeviceManager::invalidateDeviceBuffers(void *forgedTgtPtr, size_t size) {
  deviceBuffersCache.deleteEntry(forgedTgtPtr, this->api_);
  return true;
};

PJRT_Buffer *DeviceManager::moveDataSegsToDevice(
    const int devIdx, void *dataSrc, const llvm::ArrayRef<int64_t> &tensorShape,
    const llvm::ArrayRef<std::pair<size_t, size_t>> &tensorSlices,
    DType elementDType) {
  assert(tensorShape.size() == tensorSlices.size());
  auto elementSizeInByte = getDTypeSizeInByte(elementDType);
  const auto *device = pjrtDevices_[devIdx];
  llvm::SmallVector<int64_t> byteStrides(tensorShape.size());
  for (int i = byteStrides.size() - 1; i >= 0; --i) {
    byteStrides[i] = elementSizeInByte;
    if (i != byteStrides.size() - 1) {
      byteStrides[i] = tensorShape[i + 1] * byteStrides[i + 1];
    }
  }
  llvm::SmallVector<int64_t> dims(tensorShape.size());
  for (int i = 0; i < tensorShape.size(); i++) {
    dims[i] = tensorSlices[i].second - tensorSlices[i].first;
  }
  size_t offsetInByte = 0;
  for (int i = 0; i < tensorSlices.size(); i++) {
    offsetInByte += tensorSlices[i].first * byteStrides[i];
  }
  return createBufferFromForgedTgtPointers(devIdx, dataSrc, elementDType, dims,
                                           byteStrides, offsetInByte,
                                           tensorShape, ShardSlices(tensorSlices));
};

void DeviceManager::moveOutBuffersToDeviceBufferMap(
    int devIdx, void *argPtr, int argIdx, PJRT_Buffer **const *outsBuffersList,
    llvm::ArrayRef<int64_t> fullShape, const ShardSlices &slices,
    size_t elementSize) {
  assert(outsBuffersList[devIdx][argIdx] &&
         "buffer not exists in outsBuffersList!");
  DeviceBufferSlot slot{outsBuffersList[devIdx][argIdx], slices,
                        llvm::SmallVector<int64_t>(fullShape), elementSize};
  deviceBuffersCache.tryInsertOrUpdate(api_, argPtr, devIdx,
                                       pjrtDevices_.size(), std::move(slot));
}

// Copy a dense row-major shard into its place in the full row-major tensor:
// one memcpy per contiguous run along the innermost dim.
static void scatterShardToHost(const std::byte *shard, std::byte *full,
                               llvm::ArrayRef<int64_t> fullShape,
                               const ShardSlices &slices, size_t elementSize) {
  const size_t rank = fullShape.size();
  if (rank == 0) {
    std::memcpy(full, shard, elementSize);
    return;
  }
  // Strides in elements of the full tensor and of the shard.
  llvm::SmallVector<size_t> fullStride(rank, 1), shardStride(rank, 1);
  for (int dim = static_cast<int>(rank) - 2; dim >= 0; --dim) {
    fullStride[dim] = fullStride[dim + 1] * fullShape[dim + 1];
    shardStride[dim] = shardStride[dim + 1] *
                       (slices[dim + 1].second - slices[dim + 1].first);
  }
  for (size_t dim = 0; dim < rank; ++dim) {
    if (slices[dim].second <= slices[dim].first) {
      return; // empty shard
    }
  }
  const size_t runBytes =
      (slices[rank - 1].second - slices[rank - 1].first) * elementSize;

  // idx: position inside the shard over dims [0, rank - 1).
  llvm::SmallVector<size_t> idx(rank, 0);
  while (true) {
    size_t fullOffset = slices[rank - 1].first; // innermost start
    size_t shardOffset = 0;
    for (size_t dim = 0; dim + 1 < rank; ++dim) {
      fullOffset += (slices[dim].first + idx[dim]) * fullStride[dim];
      shardOffset += idx[dim] * shardStride[dim];
    }
    std::memcpy(full + fullOffset * elementSize,
                shard + shardOffset * elementSize, runBytes);

    // Advance the odometer over the outer dims.
    int dim = static_cast<int>(rank) - 2;
    for (; dim >= 0; --dim) {
      if (++idx[dim] < slices[dim].second - slices[dim].first) {
        break;
      }
      idx[dim] = 0;
    }
    if (dim < 0) {
      return;
    }
  }
}

// Read every distinct shard into a temporary dense buffer, then scatter it to
// its place in hostPtr. Devices holding the same shard (replicated axes) are
// read once.
void DeviceManager::moveDataToHostBuffer(void *hostPtr, size_t size) {
  auto buffersOptional = deviceBuffersCache.getBuffers(hostPtr);
  if (!buffersOptional.has_value()) {
    llvm::errs() << "Can not find related buffer!\n";
    return;
  }
  const std::vector<DeviceBufferSlot> &buffers = buffersOptional.value();
  if (buffers.empty() || buffers[0].bufferPtr == nullptr) {
    llvm::errs() << "Device buffer of " << hostPtr << " is missing!\n";
    std::exit(EXIT_FAILURE);
  }
  llvm::ArrayRef<int64_t> fullShape = buffers[0].fullShape;
  const size_t elementSize = buffers[0].elementSize;
  size_t fullBytes = elementSize;
  for (int64_t dimSize : fullShape) {
    fullBytes *= dimSize;
  }
  if (elementSize == 0 || fullBytes != size) {
    llvm::errs() << "Readback of " << hostPtr << ": expected " << size
                 << " bytes, the cached tensor has " << fullBytes
                 << " bytes (partial readback is not supported).\n";
    std::exit(EXIT_FAILURE);
  }

  llvm::SmallVector<const DeviceBufferSlot *> toRead;
  for (const DeviceBufferSlot &slot : buffers) {
    if (slot.bufferPtr == nullptr || slot.fullShape != buffers[0].fullShape ||
        slot.slices.size() != fullShape.size()) {
      llvm::errs() << "Device buffers of " << hostPtr
                   << " are inconsistent!\n";
      std::exit(EXIT_FAILURE);
    }
    bool seen = llvm::any_of(toRead, [&](const DeviceBufferSlot *other) {
      return other->slices == slot.slices;
    });
    if (!seen) {
      toRead.push_back(&slot);
    }
  }

  std::vector<std::vector<std::byte>> shards(toRead.size());
  llvm::SmallVector<PJRT_Event *> events;
  for (size_t i = 0; i < toRead.size(); ++i) {
    size_t shardBytes = elementSize;
    for (const auto &[start, end] : toRead[i]->slices) {
      shardBytes *= end - start;
    }
    shards[i].resize(shardBytes);
    PJRT_Buffer_ToHostBuffer_Args toHostArg = {
        .struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
        .src = toRead[i]->bufferPtr,
        .dst = shards[i].data(),
        .dst_size = shardBytes};
    auto *err = api_->PJRT_Buffer_ToHostBuffer(&toHostArg);
    if (err) {
      llvm::errs() << "Error in moving buffer to host: "
                   << JitManager::getInstance().getErrMsg(api_, err) << "\n";
      std::exit(EXIT_FAILURE);
    }
    events.push_back(toHostArg.event);
  }
  for (PJRT_Event *event : events) {
    auto awaitArgs = PJRT_Event_Await_Args{
        .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE, .event = event};
    auto *err = api_->PJRT_Event_Await(&awaitArgs);
    if (err) {
      llvm::errs() << "Error in waiting for readback: "
                   << JitManager::getInstance().getErrMsg(api_, err) << "\n";
      std::exit(EXIT_FAILURE);
    }
    PJRT_Event_Destroy_Args destroyArgs = {
        .struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE, .event = event};
    api_->PJRT_Event_Destroy(&destroyArgs);
  }

  for (size_t i = 0; i < toRead.size(); ++i) {
    scatterShardToHost(shards[i].data(), static_cast<std::byte *>(hostPtr),
                       fullShape, toRead[i]->slices, elementSize);
  }
}

void DeviceManager::destroyHostBoundBuffers(void *hostPtr) {
  this->invalidateDeviceBuffers(hostPtr, 0);
};
