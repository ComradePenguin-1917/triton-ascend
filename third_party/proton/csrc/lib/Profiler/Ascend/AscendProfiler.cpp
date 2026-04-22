#include "Profiler/Ascend/AscendProfiler.h"
#include "Driver/NPU/AscendApi.h"
#include "Driver/NPU/MsprofApi.h"
#include "Device.h"
#include "Data/Metric.h"
#include "Context/Context.h"
#include <acl/acl_prof.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <map>
#include <atomic>
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

// Global pointer to active profiler for RT callback
static std::atomic<AscendProfiler*> g_activeProfiler{nullptr};

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

// Convert kernel activity to metric
std::shared_ptr<Metric> convertKernelToMetric(uint64_t startTime, uint64_t endTime,
                                               uint32_t deviceId) {
  if (startTime < endTime) {
    return std::make_shared<KernelMetric>(
        startTime, endTime, 1, deviceId,
        static_cast<uint64_t>(DeviceType::ASCEND), 0 /* streamId */);
  }
  return nullptr;
}

} // anonymous namespace

struct AscendProfiler::AscendProfilerPimpl
    : GPUProfiler<AscendProfiler>::GPUProfilerPimplInterface {

  aclprofConfig *profConfig = nullptr;
  std::string outputPath;
  bool isStarted = false;
  bool aclInitializedByUs = false;
  bool msprofEnabled = false;

  // Kernel tracking for msprof integration
  struct KernelLaunchInfo {
    std::string name;
    uint64_t startTime;
    uint32_t deviceId;
    size_t externId;
  };

  std::map<uint64_t, KernelLaunchInfo> pendingKernels; // correlationId -> info

  // Atomic counter for generating unique correlation IDs
  std::atomic<uint64_t> nextCorrelationId{1};

  AscendProfilerPimpl(AscendProfiler &profiler)
      : GPUProfiler<AscendProfiler>::GPUProfilerPimplInterface(profiler) {}

  ~AscendProfilerPimpl() override {
    if (profConfig) {
      aclprofDestroyConfig(profConfig);
      profConfig = nullptr;
    }
  }

  // Static RT kernel callback - called from C API
  static int32_t rtKernelReportCallbackStatic(void *stream, void *kernelInfo) {
    AscendProfiler *profiler = g_activeProfiler.load();
    if (!profiler) {
      return 0;
    }

    if (auto *pImpl = dynamic_cast<AscendProfilerPimpl *>(profiler->getPimpl())) {
      pImpl->handleKernelLaunch(stream, kernelInfo);
    }

    return 0;
  }

  // Handle kernel launch callback from RT
  void handleKernelLaunch(void *stream, void *kernelInfo) {
    if (!isStarted) {
      return;
    }

    // Get timestamp
    uint64_t timestamp = msprof::getSysCycleTime<false>();

    // Generate correlation ID
    uint64_t correlationId = nextCorrelationId.fetch_add(1);

    // Get current scope from profiler's correlation tracking
    auto &correlation = this->profiler.correlation;

    if (!correlation.externIdQueue.empty()) {
      size_t externId = correlation.externIdQueue.back();

      // Record kernel launch
      KernelLaunchInfo info;
      info.name = "kernel_" + std::to_string(correlationId);
      info.startTime = timestamp;
      info.deviceId = 0; // Default device
      info.externId = externId;

      pendingKernels[correlationId] = info;

      // Correlate this kernel with the current scope
      correlation.correlate(correlationId);
      correlation.submit(correlationId);
    }
  }

  // Complete kernel execution
  void completeKernel(uint64_t correlationId, uint64_t endTime) {
    auto it = pendingKernels.find(correlationId);
    if (it == pendingKernels.end()) {
      return;
    }

    const KernelLaunchInfo &info = it->second;

    // Create metric
    auto metric = convertKernelToMetric(info.startTime, endTime, info.deviceId);
    if (metric) {
      // Add metric to all active data sets
      for (auto *data : this->profiler.getDataSet()) {
        data->addMetric(info.externId, metric);
      }
    }

    // Report to msprof if enabled
    if (g_MsprofFlagL0 || g_MsprofFlagL1) {
      reportKernelToMsprof(info, endTime);
    }

    // Mark as complete
    this->profiler.correlation.complete(correlationId);
    pendingKernels.erase(it);
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

    // 1. Register RT kernel callback for automatic kernel tracking
    g_activeProfiler.store(&this->profiler);

    typedef int32_t (*rtSetKernelReportCallbackFunc)(void *callback);
    void *rtLib = dlopen("libascendcl.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!rtLib) {
      rtLib = dlopen("libascendcl.so", RTLD_LAZY);
    }

    if (rtLib) {
      auto setCallback = reinterpret_cast<rtSetKernelReportCallbackFunc>(
          dlsym(rtLib, "rtSetKernelReportCallback"));
      if (setCallback) {
        setCallback(reinterpret_cast<void *>(&AscendProfilerPimpl::rtKernelReportCallbackStatic));
      }
    }

    // 2. Register msprof callback (CCE module = 8)
    msprofEnabled = (msprof::registerCallback<false>(8, profCtrlHandle) == 0);

    // 3. Initialize ACL Profiling
    const char *envPath = std::getenv("PROTON_ASCEND_OUTPUT_PATH");
    outputPath = (envPath && envPath[0] != '\0') ? envPath : "/tmp/ascend_profiling";

    ret = ascend::initProfiling<false>(outputPath.c_str(), outputPath.size());
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("[PROTON] ACL profiling init failed: " +
                               std::to_string(ret));
    }

    // 4. Create ACL profiling configuration with dynamic device detection
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

    uint64_t profCategory = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME;

    profConfig = aclprofCreateConfig(
        deviceIdList.data(), deviceCount,
        ACL_AICORE_NONE, nullptr,
        profCategory);

    if (!profConfig) {
      throw std::runtime_error("[PROTON] Failed to create ACL profiling config");
    }

    // 5. Start ACL profiling
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
    if (!isStarted) {
      return;
    }

    // Synchronize device to ensure all kernels are completed
    ascend::synchronizeDevice<false>(0);

    // Complete all pending kernels
    uint64_t currentTime = msprof::getSysCycleTime<false>();

    std::vector<uint64_t> completedIds;
    for (const auto &pair : pendingKernels) {
      uint64_t correlationId = pair.first;
      completeKernel(correlationId, currentTime);
      completedIds.push_back(correlationId);
    }
  }

  void doStop() override {
    if (!isStarted) {
      return;
    }

    // Flush any remaining data
    doFlush();

    // 1. Stop ACL profiling
    if (profConfig) {
      ascend::stopProfiling<false>(profConfig);
      aclprofDestroyConfig(profConfig);
      profConfig = nullptr;
    }

    // 2. Finalize ACL profiling
    ascend::finalizeProfiling<false>();

    // 3. Finalize msprof
    if (msprofEnabled) {
      msprof::finalize<false>();
    }

    // 4. Clear global profiler reference
    g_activeProfiler.store(nullptr);

    isStarted = false;
  }

  void reportKernelToMsprof(const KernelLaunchInfo &info, uint64_t endTime) {
    // Get thread ID
    uint32_t threadId = static_cast<uint32_t>(syscall(SYS_gettid));

    // Get hash ID for kernel name
    uint64_t opNameHashId = msprof::getHashId<false>(
        info.name.c_str(), info.name.length());

    // Report launch timestamp (L0)
    if (g_MsprofFlagL0 || g_MsprofFlagL1) {
      MsprofApi apiInfo{};
      apiInfo.level = MSPROF_REPORT_NODE_LEVEL;
      apiInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
      apiInfo.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      apiInfo.threadId = threadId;
      apiInfo.reserve = 0;
      apiInfo.beginTime = info.startTime;
      apiInfo.endTime = endTime;
      apiInfo.itemId = opNameHashId;

      msprof::reportApi<false>(0, &apiInfo);
    }

    // Report basic kernel info (L1)
    if (g_MsprofFlagL1) {
      MsprofCompactInfo nodeBasicInfo{};
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashId;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashId;
      nodeBasicInfo.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_AI_CORE;
      nodeBasicInfo.data.nodeBasicInfo.blockDim = 1;

      msprof::reportCompactInfo<false>(
          0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));
    }
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

uint64_t AscendProfiler::recordKernelLaunch(const char *kernelName, uint32_t gridX,
                                             uint32_t gridY, uint32_t gridZ) {
  auto *pimpl = dynamic_cast<AscendProfilerPimpl *>(pImpl.get());

  if (!pimpl) {
    return 0;
  }

  // Get current device ID from ACL
  uint32_t deviceId = 0;
  int32_t currentDevice = 0;
  aclError ret = aclrtGetDevice(&currentDevice);
  if (ret == ACL_SUCCESS && currentDevice >= 0) {
    deviceId = static_cast<uint32_t>(currentDevice);
  }

  if (!pimpl->isStarted) {
    return 0;
  }

  // Get timestamp
  uint64_t timestamp = msprof::getSysCycleTime<false>();

  // Generate correlation ID
  uint64_t correlationId = pimpl->nextCorrelationId.fetch_add(1);

  // Get current scope from profiler's correlation tracking
  auto &correlation = this->correlation;

  if (!correlation.externIdQueue.empty()) {
    size_t externId = correlation.externIdQueue.back();

    // Record kernel launch
    AscendProfilerPimpl::KernelLaunchInfo info;
    info.name = kernelName;
    info.startTime = timestamp;
    info.deviceId = deviceId;
    info.externId = externId;

    pimpl->pendingKernels[correlationId] = info;

    // Correlate this kernel with the current scope
    correlation.correlate(correlationId);
    correlation.submit(correlationId);
  }

  return correlationId;
}

void AscendProfiler::recordKernelComplete(uint64_t correlationId) {
  auto *pimpl = dynamic_cast<AscendProfilerPimpl *>(pImpl.get());
  if (!pimpl || !pimpl->isStarted || correlationId == 0) {
    return;
  }

  uint64_t endTime = msprof::getSysCycleTime<false>();
  pimpl->completeKernel(correlationId, endTime);
}

} // namespace proton
