#include "Profiler/Instrumentation/AscendRuntime.h"

#include "Driver/NPU/AscendApi.h"
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <stdexcept>
#include <algorithm>

namespace proton {

void AscendRuntime::allocateHostBuffer(uint8_t **buffer, size_t size) {
  // Allocate host memory using ACL runtime
  aclError ret = aclrtMallocHost(reinterpret_cast<void **>(buffer), size);
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to allocate host buffer: " + std::to_string(ret));
  }
}

void AscendRuntime::freeHostBuffer(uint8_t *buffer) {
  // Free host memory using ACL runtime
  aclError ret = aclrtFreeHost(buffer);
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to free host buffer: " + std::to_string(ret));
  }
}

uint64_t AscendRuntime::getDevice() {
  // Get current device ID
  int32_t deviceId;
  aclError ret = aclrtGetDevice(&deviceId);
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to get device: " + std::to_string(ret));
  }
  return static_cast<uint64_t>(deviceId);
}

void *AscendRuntime::getPriorityStream() {
  // Create a stream with priority
  // Note: Ascend uses aclrtStream which is similar to CUDA streams
  aclrtStream stream;
  // ACL_RT_PRIORITY_0 is the highest priority
  aclError ret = aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_FAST_LAUNCH);
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to create stream: " + std::to_string(ret));
  }
  return reinterpret_cast<void *>(stream);
}

void AscendRuntime::synchronizeStream(void *stream) {
  // Synchronize the stream
  aclError ret = aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to synchronize stream: " + std::to_string(ret));
  }
}

void AscendRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {
  // Process buffer by copying from device to host in chunks
  int64_t chunkSize = std::min(hostBufferSize, deviceBufferSize);
  int64_t sizeLeftOnDevice = deviceBufferSize;
  
  while (chunkSize > 0) {
    // Copy data from device to host asynchronously
    aclError ret = aclrtMemcpyAsync(
        reinterpret_cast<void *>(hostBuffer),
        chunkSize,
        reinterpret_cast<void *>(deviceBuffer),
        chunkSize,
        ACL_MEMCPY_DEVICE_TO_HOST,
        reinterpret_cast<aclrtStream>(stream));
    
    if (ret != ACL_ERROR_NONE) {
      throw std::runtime_error("Failed to copy memory from device to host: " + std::to_string(ret));
    }
    
    // Synchronize to ensure the copy is complete before processing
    ret = aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_ERROR_NONE) {
      throw std::runtime_error("Failed to synchronize stream during buffer processing: " + std::to_string(ret));
    }
    
    // Process the copied data
    callback(hostBuffer, chunkSize);
    
    // Update for next chunk
    sizeLeftOnDevice -= chunkSize;
    deviceBuffer += chunkSize;
    chunkSize = std::min(static_cast<int64_t>(hostBufferSize), sizeLeftOnDevice);
  }
}

} // namespace proton
