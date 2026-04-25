#include "Profiler/Ascend/AscendProfiler.h"
#include "Profiler/Ascend/AclProfParser.h"
#include "Driver/NPU/AscendApi.h"
#include "Driver/NPU/MsprofApi.h"
#include "Device.h"
#include "Data/Metric.h"
#include "Context/Context.h"
#include <acl/acl_prof.h>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace proton {

// Explicit instantiation of static member variables
template <>
thread_local GPUProfiler<AscendProfiler>::ThreadState
    GPUProfiler<AscendProfiler>::threadState(AscendProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<AscendProfiler>::Correlation::externIdQueue{};

namespace {

// Static flags for profiling control
static uint32_t g_MsprofFlagL0 = 0;
static uint32_t g_MsprofFlagL1 = 0;

// Profiling control callback for msprof
int32_t profCtrlHandle(uint32_t ctrlType, void *ctrlData, uint32_t dataLen) {
  if ((ctrlData == nullptr) || (dataLen == 0U)) {
    return 1;
  }

  if (ctrlType == 1) {
    MsprofCommandHandle *handle = static_cast<MsprofCommandHandle *>(ctrlData);
    if (handle->type >= 6) // 6 is not used
      return 1;
    if (handle->type == 1) { // init - 0, start - 1
      g_MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
      g_MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
    }
  }
  return 0;
}

} // anonymous namespace

struct AscendProfiler::AscendProfilerPimpl
    : GPUProfiler<AscendProfiler>::GPUProfilerPimplInterface {

  aclprofConfig *profConfig = nullptr;
  std::string outputPath;
  bool isStarted = false;
  bool aclInitializedByUs = false;
  bool msprofEnabled = false;
  std::vector<Data *> dataPtrs;

  AscendProfilerPimpl(AscendProfiler &profiler)
      : GPUProfiler<AscendProfiler>::GPUProfilerPimplInterface(profiler) {}

  ~AscendProfilerPimpl() override {
    if (profConfig) {
      aclprofDestroyConfig(profConfig);
      profConfig = nullptr;
    }
  }

  void setLibPath(const std::string &libPath) override {
    // For Ascend, we don't need to dynamically load libraries
    // ACL and msprof are linked statically or found via standard paths
    // This method is here to satisfy the pure virtual interface
  }

  void doStart() override {
    if (isStarted) {
      return;
    }

    // 0. Ensure ACL runtime is initialized
    aclError ret = aclInit(nullptr);
    if (ret == ACL_SUCCESS) {
      aclInitializedByUs = true;
    } else if (ret == ACL_ERROR_REPEAT_INITIALIZE) {
      aclInitializedByUs = false;
    } else {
      throw std::runtime_error("[PROTON] Failed to initialize ACL runtime: aclInit returned " +
                               std::to_string(ret));
    }

    // 1. Register msprof callback (CCE module = 8)
    msprofEnabled = (msprof::registerCallback<false>(8, profCtrlHandle) == 0);

    // 2. Initialize ACL Profiling
    const char *envPath = std::getenv("PROTON_ASCEND_OUTPUT_PATH");
    outputPath = (envPath && envPath[0] != '\0') ? envPath : "/tmp/ascend_profiling";

    ret = ascend::initProfiling<false>(outputPath.c_str(), outputPath.size());
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("[PROTON] ACL profiling init failed: " +
                               std::to_string(ret));
    }

    // 3. Create ACL profiling configuration with dynamic device detection
    uint32_t deviceCount = 0;
    ret = aclrtGetDeviceCount(&deviceCount);
    if (ret != ACL_SUCCESS || deviceCount == 0) {
      throw std::runtime_error("[PROTON] Failed to get device count: aclrtGetDeviceCount returned " +
                               std::to_string(ret));
    }

    std::vector<uint32_t> deviceIdList(deviceCount);
    for (uint32_t i = 0; i < deviceCount; ++i) {
      deviceIdList[i] = i;
    }

    uint64_t profCategory = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS;

    profConfig = aclprofCreateConfig(
        deviceIdList.data(), deviceCount,
        ACL_AICORE_PIPE_UTILIZATION, nullptr,
        profCategory);

    if (!profConfig) {
      throw std::runtime_error("[PROTON] Failed to create ACL profiling config");
    }

    // 4. Start ACL profiling
    ret = ascend::startProfiling<false>(profConfig);
    if (ret != ACL_SUCCESS) {
      aclprofDestroyConfig(profConfig);
      profConfig = nullptr;
      throw std::runtime_error("[PROTON] Failed to start ACL profiling: " +
                               std::to_string(ret));
    }

    isStarted = true;
  }

  void doFlush() override {
    if (!isStarted)
      return;
    ascend::synchronizeDevice<false>(0);
    auto ds = this->profiler.getDataSet();
    if (!ds.empty()) {
      dataPtrs.clear();
      for (auto *data : ds)
        dataPtrs.push_back(data);
    }
  }

  void doStop() override {
    if (!isStarted) {
      return;
    }
    doFlush();

    if (profConfig) {
      ascend::stopProfiling<false>(profConfig);
      aclprofDestroyConfig(profConfig);
      profConfig = nullptr;
    }

    auto kernelEntries = AclProfParser::parseProfilingData(outputPath);
    for (const auto &entry : kernelEntries) {
      uint64_t endTimeNs = entry.startTimeNs + entry.durationUs * 1000;
      auto metric = std::make_shared<KernelMetric>(
          entry.startTimeNs, endTimeNs, 1, entry.deviceId,
          static_cast<uint64_t>(DeviceType::ASCEND), entry.streamId);

      for (auto *data : dataPtrs) {
        size_t kernelScopeId = Scope::getNewScopeId();
        size_t opScopeId = data->addOp(kernelScopeId, entry.opName);
        data->addMetric(opScopeId, metric);
        if (entry.aicMacRatio > 0)
          data->addMetrics(opScopeId, {{"aic_mac_ratio", entry.aicMacRatio}});
        if (entry.aicScalarRatio > 0)
          data->addMetrics(opScopeId, {{"aic_scalar_ratio", entry.aicScalarRatio}});
        if (entry.aicMte1Ratio > 0)
          data->addMetrics(opScopeId, {{"aic_mte1_ratio", entry.aicMte1Ratio}});
        if (entry.aicMte2Ratio > 0)
          data->addMetrics(opScopeId, {{"aic_mte2_ratio", entry.aicMte2Ratio}});
        if (entry.aicFixpipeRatio > 0)
          data->addMetrics(opScopeId, {{"aic_fixpipe_ratio", entry.aicFixpipeRatio}});
        if (entry.aicIcacheMissRate > 0)
          data->addMetrics(opScopeId, {{"aic_icache_miss_rate", entry.aicIcacheMissRate}});
        if (entry.aivVecRatio > 0)
          data->addMetrics(opScopeId, {{"aiv_vec_ratio", entry.aivVecRatio}});
        if (entry.aivScalarRatio > 0)
          data->addMetrics(opScopeId, {{"aiv_scalar_ratio", entry.aivScalarRatio}});
        if (entry.aivMte1Ratio > 0)
          data->addMetrics(opScopeId, {{"aiv_mte1_ratio", entry.aivMte1Ratio}});
        if (entry.aivMte2Ratio > 0)
          data->addMetrics(opScopeId, {{"aiv_mte2_ratio", entry.aivMte2Ratio}});
        if (entry.aivMte3Ratio > 0)
          data->addMetrics(opScopeId, {{"aiv_mte3_ratio", entry.aivMte3Ratio}});
        if (entry.aivIcacheMissRate > 0)
          data->addMetrics(opScopeId, {{"aiv_icache_miss_rate", entry.aivIcacheMissRate}});
      }
    }

    ascend::finalizeProfiling<false>();

    if (msprofEnabled) {
      msprof::finalize<false>();
    }

    isStarted = false;
  }

};

AscendProfiler::AscendProfiler() {
  pImpl = std::make_unique<AscendProfilerPimpl>(*this);
}

AscendProfiler::~AscendProfiler() = default;

void AscendProfiler::doSetMode(const std::vector<std::string> &modeAndOptions) {
  // For Ascend, we don't need to do anything special in setMode
  // The mode is implicitly "ascend" and msprof handles profiling configuration
  // This implementation satisfies the pure virtual method requirement
}

void AscendProfiler::startOp(const Scope &scope) {
  GPUProfiler<AscendProfiler>::startOp(scope);
  scopeStartTimes_[scope.scopeId] = msprof::getSysCycleTime<false>();
}

void AscendProfiler::stopOp(const Scope &scope) {
  GPUProfiler<AscendProfiler>::stopOp(scope);
  auto it = scopeStartTimes_.find(scope.scopeId);
  if (it == scopeStartTimes_.end())
    return;
  uint64_t startTime = it->second;
  uint64_t endTime = msprof::getSysCycleTime<false>();
  scopeStartTimes_.erase(it);

  if (endTime <= startTime)
    return;

  uint32_t deviceId = 0;
  int32_t currentDevice = 0;
  if (aclrtGetDevice(&currentDevice) == ACL_SUCCESS && currentDevice >= 0)
    deviceId = static_cast<uint32_t>(currentDevice);

  auto metric = std::make_shared<KernelMetric>(
      startTime, endTime, 1, deviceId,
      static_cast<uint64_t>(DeviceType::ASCEND), 0);

  for (auto *data : getDataSet())
    data->addMetric(scope.scopeId, metric);
}

void AscendProfiler::enterScope(const Scope &scope) {
  scopeStartTimes_[scope.scopeId] = msprof::getSysCycleTime<false>();
}

void AscendProfiler::exitScope(const Scope &scope) {
  auto it = scopeStartTimes_.find(scope.scopeId);
  if (it == scopeStartTimes_.end())
    return;

  uint64_t startTime = it->second;
  uint64_t endTime = msprof::getSysCycleTime<false>();
  scopeStartTimes_.erase(it);

  if (endTime <= startTime)
    return;

  uint32_t deviceId = 0;
  int32_t currentDevice = 0;
  if (aclrtGetDevice(&currentDevice) == ACL_SUCCESS && currentDevice >= 0)
    deviceId = static_cast<uint32_t>(currentDevice);

  auto metric = std::make_shared<KernelMetric>(
      startTime, endTime, 1, deviceId,
      static_cast<uint64_t>(DeviceType::ASCEND), 0);

  for (auto *data : getDataSet()) {
    size_t opScopeId = data->addOp(scope.scopeId, scope.name);
    data->addMetric(opScopeId, metric);
  }
}

} // namespace proton
