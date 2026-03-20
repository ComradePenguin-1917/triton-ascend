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
#include <iostream>
#include <memory>
#include <map>
#include <atomic>

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
    if (handle->type >= 6)  // 6 is not used
      return 1;
    if (handle->type == 1) {  // init - 0, start - 1
      g_MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
      g_MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      std::cerr << "[AscendProfiler] msprof profiling started: L0="
                << g_MsprofFlagL0 << ", L1=" << g_MsprofFlagL1 << std::endl;
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

  std::map<uint64_t, KernelLaunchInfo> pendingKernels;  // correlationId -> info

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

    // Access pImpl through friend class or public method
    // We'll add a public method to AscendProfiler to get pImpl
    if (auto *pImpl = dynamic_cast<AscendProfilerPimpl*>(profiler->getPimpl())) {
      pImpl->handleKernelLaunch(stream, kernelInfo);
    }

    return 0;
  }

  // Handle kernel launch callback from RT
  void handleKernelLaunch(void *stream, void *kernelInfo) {
    std::cerr << "[AscendProfiler::handleKernelLaunch] CALLBACK INVOKED! stream="
              << stream << ", kernelInfo=" << kernelInfo << std::endl;

    if (!isStarted) {
      std::cerr << "[AscendProfiler::handleKernelLaunch] Profiling not started, ignoring" << std::endl;
      return;
    }

    // Get timestamp
    uint64_t timestamp = msprof::getSysCycleTime<false>();

    // Generate correlation ID
    uint64_t correlationId = nextCorrelationId.fetch_add(1);

    // Get current scope from profiler's correlation tracking
    // This links the kernel to the current Python scope
    auto &correlation = this->profiler.correlation;

    std::cerr << "[AscendProfiler::handleKernelLaunch] externIdQueue.size()="
              << correlation.externIdQueue.size() << std::endl;

    if (!correlation.externIdQueue.empty()) {
      size_t externId = correlation.externIdQueue.back();

      // Record kernel launch
      KernelLaunchInfo info;
      info.name = "kernel_" + std::to_string(correlationId);  // Will be replaced with actual name if available
      info.startTime = timestamp;
      info.deviceId = 0;  // Default device
      info.externId = externId;

      pendingKernels[correlationId] = info;

      // Correlate this kernel with the current scope
      correlation.correlate(correlationId);
      correlation.submit(correlationId);

      std::cerr << "[AscendProfiler::handleKernelLaunch] Recorded kernel launch: correlationId="
                << correlationId << ", externId=" << externId << ", timestamp=" << timestamp
                << ", pendingKernels.size()=" << pendingKernels.size() << std::endl;
    } else {
      std::cerr << "[AscendProfiler::handleKernelLaunch] WARNING: externIdQueue is empty, "
                << "kernel not recorded!" << std::endl;
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

      std::cerr << "[AscendProfiler::completeKernel] Added metric: correlationId="
                << correlationId << ", duration=" << (endTime - info.startTime) << std::endl;
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

    std::cerr << "[AscendProfiler] Starting profiling..." << std::endl;

    // 0. Ensure ACL runtime is initialized
    aclError ret = aclInit(nullptr);
    if (ret == ACL_SUCCESS) {
      aclInitializedByUs = true;
      std::cerr << "[AscendProfiler] ACL initialized by profiler" << std::endl;
    } else if (ret == ACL_ERROR_REPEAT_INITIALIZE) {
      aclInitializedByUs = false;
      std::cerr << "[AscendProfiler] ACL already initialized" << std::endl;
    } else {
      throw std::runtime_error("Failed to initialize ACL runtime: aclInit returned " +
                               std::to_string(ret));
    }

    // 1. Register RT kernel callback for automatic kernel tracking
    // This is the CUDA CUPTI equivalent for Ascend!
    g_activeProfiler.store(&this->profiler);

    // Note: rtSetKernelReportCallback is from runtime/kernel.h
    // We need to declare it here since we're using it dynamically
    typedef int32_t (*rtSetKernelReportCallbackFunc)(void* callback);
    void *rtLib = dlopen("libascendcl.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!rtLib) {
      rtLib = dlopen("libascendcl.so", RTLD_LAZY);
    }

    if (rtLib) {
      auto setCallback = reinterpret_cast<rtSetKernelReportCallbackFunc>(
          dlsym(rtLib, "rtSetKernelReportCallback"));
      if (setCallback) {
        int32_t cbRet = setCallback(reinterpret_cast<void*>(&AscendProfilerPimpl::rtKernelReportCallbackStatic));
        if (cbRet == 0) {
          std::cerr << "[AscendProfiler] RT kernel callback registered successfully" << std::endl;
        } else {
          std::cerr << "[AscendProfiler] Warning: RT kernel callback registration returned "
                    << cbRet << std::endl;
        }
      } else {
        std::cerr << "[AscendProfiler] Warning: rtSetKernelReportCallback not found" << std::endl;
      }
    } else {
      std::cerr << "[AscendProfiler] Warning: Could not load libascendcl.so for RT callbacks" << std::endl;
    }

    // 2. Register msprof callback (CCE module = 8)
    // This allows msprof to collect kernel performance data
    msprofEnabled = (msprof::registerCallback<false>(8, profCtrlHandle) == 0);
    if (msprofEnabled) {
      std::cerr << "[AscendProfiler] msprof callback registered successfully" << std::endl;
    } else {
      std::cerr << "[AscendProfiler] Warning: msprof callback registration failed, "
                << "continuing with ACL profiling only" << std::endl;
    }

    // 3. Initialize ACL Profiling (for backup/additional data)
    outputPath = "/tmp/ascend_profiling";
    ret = ascend::initProfiling<false>(outputPath.c_str(), outputPath.size());
    if (ret != ACL_SUCCESS) {
      std::cerr << "[AscendProfiler] Warning: ACL profiling init failed: "
                << ret << std::endl;
    } else {
      std::cerr << "[AscendProfiler] ACL profiling initialized" << std::endl;
    }

    // 3. Create ACL profiling configuration
    uint32_t deviceIdList[] = {0};
    uint32_t deviceNums = 1;
    uint64_t profCategory = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME;

    profConfig = aclprofCreateConfig(
        deviceIdList, deviceNums,
        ACL_AICORE_NONE, nullptr,
        profCategory);

    if (!profConfig) {
      std::cerr << "[AscendProfiler] Warning: Failed to create ACL profiling config"
                << std::endl;
    } else {
      // 4. Start ACL profiling
      ret = ascend::startProfiling<false>(profConfig);
      if (ret != ACL_SUCCESS) {
        std::cerr << "[AscendProfiler] Warning: Failed to start ACL profiling: "
                  << ret << std::endl;
        aclprofDestroyConfig(profConfig);
        profConfig = nullptr;
      } else {
        std::cerr << "[AscendProfiler] ACL profiling started" << std::endl;
      }
    }

    isStarted = true;
    std::cerr << "[AscendProfiler] Profiling started successfully" << std::endl;
  }

  void doFlush() override {
    if (!isStarted) {
      return;
    }

    std::cerr << "[AscendProfiler] Flushing profiling data..." << std::endl;
    std::cerr << "[AscendProfiler] Current pendingKernels.size() = "
              << pendingKernels.size() << std::endl;

    // Synchronize device to ensure all kernels are completed
    ascend::synchronizeDevice<false>(0);

    // Complete all pending kernels
    // Get current timestamp as end time for all pending kernels
    uint64_t currentTime = msprof::getSysCycleTime<false>();

    std::vector<uint64_t> completedIds;
    for (const auto &pair : pendingKernels) {
      uint64_t correlationId = pair.first;
      completeKernel(correlationId, currentTime);
      completedIds.push_back(correlationId);
    }

    std::cerr << "[AscendProfiler] Completed " << completedIds.size()
              << " pending kernels during flush" << std::endl;

    std::cerr << "[AscendProfiler] Flush completed" << std::endl;
  }

  void doStop() override {
    if (!isStarted) {
      return;
    }

    std::cerr << "[AscendProfiler] Stopping profiling..." << std::endl;

    // Flush any remaining data
    doFlush();

    // 1. Stop ACL profiling
    if (profConfig) {
      aclError ret = ascend::stopProfiling<false>(profConfig);
      if (ret != ACL_SUCCESS) {
        std::cerr << "[AscendProfiler] Warning: aclprofStop returned " << ret << std::endl;
      }
      aclprofDestroyConfig(profConfig);
      profConfig = nullptr;
    }

    // 2. Finalize ACL profiling
    aclError ret = ascend::finalizeProfiling<false>();
    if (ret != ACL_SUCCESS) {
      std::cerr << "[AscendProfiler] Warning: aclprofFinalize returned " << ret << std::endl;
    }

    // 3. Finalize msprof
    if (msprofEnabled) {
      msprof::finalize<false>();
    }

    // 4. Clear global profiler reference
    g_activeProfiler.store(nullptr);

    isStarted = false;
    std::cerr << "[AscendProfiler] Profiling stopped" << std::endl;
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
      nodeBasicInfo.data.nodeBasicInfo.blockDim = 1;  // Default to 1

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

uint64_t AscendProfiler::recordKernelLaunch(const char* kernelName, uint32_t gridX,
                                             uint32_t gridY, uint32_t gridZ) {
  auto *pimpl = dynamic_cast<AscendProfilerPimpl*>(pImpl.get());

  std::cerr << "[AscendProfiler::recordKernelLaunch] Called from launcher: kernelName="
            << kernelName << ", pimpl=" << (void*)pimpl;

  if (!pimpl) {
    std::cerr << " - pimpl is NULL!" << std::endl;
    return 0;
  }

  // Get current device ID from ACL
  uint32_t deviceId = 0;
  int32_t currentDevice = 0;
  aclError ret = aclrtGetDevice(&currentDevice);
  if (ret == ACL_SUCCESS && currentDevice >= 0) {
    deviceId = static_cast<uint32_t>(currentDevice);
  }

  std::cerr << ", isStarted=" << pimpl->isStarted << ", deviceId=" << deviceId << std::endl;

  if (!pimpl->isStarted) {
    std::cerr << "[AscendProfiler::recordKernelLaunch] Profiler not started, ignoring" << std::endl;
    return 0;
  }

  // Get timestamp
  uint64_t timestamp = msprof::getSysCycleTime<false>();

  // Generate correlation ID
  uint64_t correlationId = pimpl->nextCorrelationId.fetch_add(1);

  // Get current scope from profiler's correlation tracking
  auto &correlation = this->correlation;

  std::cerr << "[AscendProfiler::recordKernelLaunch] Called from launcher: kernelName="
            << kernelName << ", correlationId=" << correlationId
            << ", externIdQueue.size()=" << correlation.externIdQueue.size() << std::endl;

  if (!correlation.externIdQueue.empty()) {
    size_t externId = correlation.externIdQueue.back();

    // Record kernel launch
    AscendProfilerPimpl::KernelLaunchInfo info;
    info.name = kernelName;
    info.startTime = timestamp;
    info.deviceId = deviceId;  // Use auto-detected device ID from ACL
    info.externId = externId;

    pimpl->pendingKernels[correlationId] = info;

    // Correlate this kernel with the current scope
    correlation.correlate(correlationId);
    correlation.submit(correlationId);

    std::cerr << "[AscendProfiler::recordKernelLaunch] Recorded kernel: correlationId="
              << correlationId << ", externId=" << externId << ", timestamp=" << timestamp
              << ", pendingKernels.size()=" << pimpl->pendingKernels.size() << std::endl;
  } else {
    std::cerr << "[AscendProfiler::recordKernelLaunch] WARNING: externIdQueue is empty, "
              << "kernel not recorded!" << std::endl;
  }

  return correlationId;
}

void AscendProfiler::recordKernelComplete(uint64_t correlationId) {
  auto *pimpl = dynamic_cast<AscendProfilerPimpl*>(pImpl.get());
  if (!pimpl || !pimpl->isStarted || correlationId == 0) {
    return;
  }

  uint64_t endTime = msprof::getSysCycleTime<false>();
  pimpl->completeKernel(correlationId, endTime);
}

} // namespace proton
