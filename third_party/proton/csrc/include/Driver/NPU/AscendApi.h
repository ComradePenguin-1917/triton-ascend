#ifndef PROTON_DRIVER_NPU_ASCEND_API_H_
#define PROTON_DRIVER_NPU_ASCEND_API_H_

#include <acl/acl.h>
#include <acl/acl_prof.h>
#include <stdexcept>
#include <string>

// Forward declare Device from parent namespace
namespace proton {
struct Device;
}

namespace proton {
namespace ascend {

// ACL Profiling 初始化
template <bool CheckSuccess>
inline aclError initProfiling(const char *profilerResultPath, size_t pathLen) {
  aclError ret = aclprofInit(profilerResultPath, pathLen);
  if constexpr (CheckSuccess) {
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("aclprofInit failed: " + std::to_string(ret));
    }
  }
  return ret;
}

// 启动 Profiling
template <bool CheckSuccess>
inline aclError startProfiling(const aclprofConfig *profilerConfig) {
  aclError ret = aclprofStart(profilerConfig);
  if constexpr (CheckSuccess) {
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("aclprofStart failed: " + std::to_string(ret));
    }
  }
  return ret;
}

// 停止 Profiling
template <bool CheckSuccess>
inline aclError stopProfiling(const aclprofConfig *profilerConfig) {
  aclError ret = aclprofStop(profilerConfig);
  if constexpr (CheckSuccess) {
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("aclprofStop failed: " + std::to_string(ret));
    }
  }
  return ret;
}

// Finalize Profiling
template <bool CheckSuccess>
inline aclError finalizeProfiling() {
  aclError ret = aclprofFinalize();
  if constexpr (CheckSuccess) {
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("aclprofFinalize failed: " +
                               std::to_string(ret));
    }
  }
  return ret;
}

// 同步当前 Stream
template <bool CheckSuccess>
inline aclError synchronizeStream(aclrtStream stream) {
  if (!stream) {
    return ACL_SUCCESS; // No stream to synchronize
  }
  aclError ret = aclrtSynchronizeStream(stream);
  if constexpr (CheckSuccess) {
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("aclrtSynchronizeStream failed: " +
                               std::to_string(ret));
    }
  }
  return ret;
}

// Note: ACL does not have aclrtGetCurrentStream
// Applications need to manage and track their own streams
// For profiling, we can use device synchronization instead
template <bool CheckSuccess>
inline aclError synchronizeDevice(int32_t deviceId) {
  aclError ret = aclrtSynchronizeDevice();
  if constexpr (CheckSuccess) {
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("aclrtSynchronizeDevice failed: " +
                               std::to_string(ret));
    }
  }
  return ret;
}

// Get device information (returns proton::Device, not ascend::Device)
proton::Device getDevice(uint64_t index);

} // namespace ascend
} // namespace proton

#endif // PROTON_DRIVER_NPU_ASCEND_API_H_
