#include "Device.h"
#ifdef PROTON_ENABLE_CUDA
#include "Driver/GPU/CudaApi.h"
#endif
#ifdef PROTON_ENABLE_HIP
#include "Driver/GPU/HipApi.h"
#endif
#ifdef PROTON_ENABLE_NPU
#include "Driver/NPU/AscendApi.h"
#endif

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
#ifdef PROTON_ENABLE_CUDA
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  }
#endif
#ifdef PROTON_ENABLE_HIP
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
#endif
#ifdef PROTON_ENABLE_NPU
  if (type == DeviceType::ASCEND) {
    return ascend::getDevice(index);
  }
#endif
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
#ifdef PROTON_ENABLE_CUDA
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  }
#endif
#ifdef PROTON_ENABLE_HIP
  if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  }
#endif
#ifdef PROTON_ENABLE_NPU
  if (type == DeviceType::ASCEND) {
    return DeviceTraits<DeviceType::ASCEND>::name;
  }
#endif
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
