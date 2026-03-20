#include "Driver/NPU/AscendApi.h"
#include "Device.h"
#include <acl/acl.h>
#include <cstring>

namespace proton {
namespace ascend {

proton::Device getDevice(uint64_t index) {
  // Query device properties using ACL APIs
  aclError ret;
  
  // Get SOC version (architecture name)
  const char *socVersion = aclrtGetSocName();
  std::string arch = socVersion ? std::string(socVersion) : "unknown";
  
  // Get device count to validate index
  uint32_t deviceCount = 0;
  ret = aclrtGetDeviceCount(&deviceCount);
  if (ret != ACL_SUCCESS || index >= deviceCount) {
    // Return a basic device if query fails
    return proton::Device(proton::DeviceType::ASCEND, index, 0, 0, 0, 0, arch);
  }
  
  // Set device to query its properties
  ret = aclrtSetDevice(index);
  if (ret != ACL_SUCCESS) {
    return proton::Device(proton::DeviceType::ASCEND, index, 0, 0, 0, 0, arch);
  }
  
  // ============================================================================
  // Ascend Device Information Strategy:
  // ----------------------------------------------------------------------------
  // Unlike CUDA/HIP, Ascend does not provide APIs to query static hardware specs
  // (clock rates, bus width, etc.). Instead, we use:
  // 
  // 1. ACL Runtime APIs - For queryable device properties
  // 2. SOC Architecture - For AI Core count inference
  // 3. MsProf Runtime - For dynamic performance metrics during profiling
  //
  // Legacy CUDA fields (clockRate, memoryClockRate, busWidth) are set to 0.
  // Ascend-specific metrics are stored in extendedProps for runtime profiling.
  // ============================================================================
  
  // Query device memory information via ACL
  size_t freeMemory = 0;
  size_t totalMemory = 0;
  ret = aclrtGetMemInfo(ACL_HBM_MEM, &freeMemory, &totalMemory);
  
  // Query device utilization (instantaneous, for monitoring)
  aclrtUtilizationInfo utilizationInfo = {0};
  aclError utilRet = aclrtGetDeviceUtilizationRate(index, &utilizationInfo);
  
  // Legacy CUDA/HIP fields - Not applicable for Ascend
  uint64_t clockRate = 0;          // Not available via ACL API
  uint64_t memoryClockRate = 0;    // Not available via ACL API  
  uint64_t busWidth = 0;            // Not available via ACL API
  uint64_t numSms = 0;              // AI Core count (inferred from SOC)
  
  // ----------------------------------------------------------------------------
  // Infer AI Core count from SOC architecture
  // ----------------------------------------------------------------------------
  if (arch.find("910") != std::string::npos) {
    numSms = 32;  // Ascend 910 series: all variants have 32 AI Cores
  } else if (arch.find("310P") != std::string::npos || 
             arch.find("310p") != std::string::npos) {
    numSms = 8;   // Ascend 310P series: 8 AI Cores
  } else if (arch.find("310") != std::string::npos) {
    numSms = 8;   // Ascend 310: 8 AI Cores
  }
  
  // Create device with legacy fields
  Device device(DeviceType::ASCEND, index, clockRate, memoryClockRate, 
                busWidth, numSms, arch);
  
  // ----------------------------------------------------------------------------
  // Note: Ascend-specific metrics are collected at runtime by MsProf
  // Including: op_name, op_type, task_duration_us, block_dim, 
  // current_freq_mhz, rated_freq_mhz, arithmetic_utilization,
  // memory_bandwidth_gbps, l0_buffer_usage, etc.
  // ----------------------------------------------------------------------------
  
  // ----------------------------------------------------------------------------
  // Runtime profiling metrics (collected by MsProf during execution):
  // 
  // Available via --aic-metrics:
  //   - ArithmeticUtilization: CUBE/Vector/Scalar unit usage
  //   - PipeUtilization: Pipeline efficiency
  //   - Memory: HBM/DDR bandwidth (GB/s)
  //   - MemoryL0: L0 Buffer read/write counts
  //   - MemoryUB: Unified Buffer usage
  //   - L2Cache: L2 cache hit rate, bandwidth
  //   - ResourceConflictRatio: Bank/port conflicts
  //   - MemoryAccess: Access patterns (sequential/random)
  //
  // Per-task metrics (task-based mode):
  //   - op_name: Operator name (e.g., "MatMul", "Conv2D")
  //   - op_type: Operator type/category
  //   - task_duration_us: Task execution time (microseconds)
  //   - task_start_time_ns: Start timestamp (nanoseconds)
  //   - task_end_time_ns: End timestamp (nanoseconds)
  //   - block_dim: Thread block dimensions
  //   - grid_dim: Grid dimensions
  //   - device_id: NPU device ID
  //   - stream_id: Stream ID
  //   - pid: Process ID
  //
  // Hardware metrics (via --sys-hardware-mem):
  //   - current_freq_mhz: Current AI Core frequency
  //   - rated_freq_mhz: Rated/maximum AI Core frequency
  //   - hbm_read_bandwidth_gbps: HBM read bandwidth
  //   - hbm_write_bandwidth_gbps: HBM write bandwidth
  //   - ddr_read_bandwidth_gbps: DDR read bandwidth
  //   - ddr_write_bandwidth_gbps: DDR write bandwidth
  //
  // See docs/MSPROF_METRICS.md for complete list
  // ----------------------------------------------------------------------------
  
  return device;
}

} // namespace ascend
} // namespace proton
