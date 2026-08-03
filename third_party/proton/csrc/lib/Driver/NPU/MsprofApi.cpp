#include "Driver/NPU/MsprofApi.h"
#include "Utility/Errors.h"

#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace proton {
namespace msprof {

namespace {

// Function pointers for dynamically loaded msprof functions
typedef int32_t (*MsprofRegisterCallbackFunc)(uint32_t, ProfCommandHandle);
typedef int32_t (*MsprofReportApiFunc)(uint32_t, const struct MsprofApi *);
typedef int32_t (*MsprofReportCompactInfoFunc)(uint32_t, const void *, uint32_t);
typedef int32_t (*MsprofReportAdditionalInfoFunc)(uint32_t, const void *, uint32_t);
typedef uint64_t (*MsprofGetHashIdFunc)(const char *, size_t);
typedef uint64_t (*MsprofSysCycleTimeFunc)();
typedef int32_t (*MsprofInitFunc)(uint32_t, void *, uint32_t);

// Global function pointers
MsprofRegisterCallbackFunc g_MsprofRegisterCallback = nullptr;
MsprofReportApiFunc g_MsprofReportApi = nullptr;
MsprofReportCompactInfoFunc g_MsprofReportCompactInfo = nullptr;
MsprofReportAdditionalInfoFunc g_MsprofReportAdditionalInfo = nullptr;
MsprofGetHashIdFunc g_MsprofGetHashId = nullptr;
MsprofSysCycleTimeFunc g_MsprofSysCycleTime = nullptr;
MsprofInitFunc g_MsprofInit = nullptr;

void *g_msprofHandle = nullptr;
std::once_flag g_initFlag;
bool g_initialized = false;

void initMsprofLibrary() {
  std::call_once(g_initFlag, []() {
    // Try to load the msprof profapi library (correct library for msprof APIs)
    const char *msprofLib = "libprofapi.so";
    g_msprofHandle = dlopen(msprofLib, RTLD_LAZY | RTLD_LOCAL);
    
    if (!g_msprofHandle) {
      std::cerr << "[MsprofApi] Warning: Could not load " << msprofLib 
                << ": " << dlerror() << std::endl;
      std::cerr << "[MsprofApi] msprof profiling will be disabled" << std::endl;
      return;
    }

    // Load function pointers
    g_MsprofRegisterCallback = reinterpret_cast<MsprofRegisterCallbackFunc>(
        dlsym(g_msprofHandle, "MsprofRegisterCallback"));
    g_MsprofReportApi = reinterpret_cast<MsprofReportApiFunc>(
        dlsym(g_msprofHandle, "MsprofReportApi"));
    g_MsprofReportCompactInfo = reinterpret_cast<MsprofReportCompactInfoFunc>(
        dlsym(g_msprofHandle, "MsprofReportCompactInfo"));
    g_MsprofReportAdditionalInfo = reinterpret_cast<MsprofReportAdditionalInfoFunc>(
        dlsym(g_msprofHandle, "MsprofReportAdditionalInfo"));
    g_MsprofGetHashId = reinterpret_cast<MsprofGetHashIdFunc>(
        dlsym(g_msprofHandle, "MsprofGetHashId"));
    g_MsprofSysCycleTime = reinterpret_cast<MsprofSysCycleTimeFunc>(
        dlsym(g_msprofHandle, "MsprofSysCycleTime"));
    g_MsprofInit = reinterpret_cast<MsprofInitFunc>(
        dlsym(g_msprofHandle, "MsprofInit"));

    // Check if all required functions were loaded
    if (g_MsprofRegisterCallback && g_MsprofReportApi && 
        g_MsprofReportCompactInfo && g_MsprofReportAdditionalInfo &&
        g_MsprofGetHashId && g_MsprofSysCycleTime) {
      g_initialized = true;
      std::cerr << "[MsprofApi] Successfully initialized msprof library" << std::endl;
    } else {
      std::cerr << "[MsprofApi] Warning: Some msprof functions could not be loaded" << std::endl;
      if (g_msprofHandle) {
        dlclose(g_msprofHandle);
        g_msprofHandle = nullptr;
      }
    }
  });
}

bool isMsprofAvailable() {
  initMsprofLibrary();
  return g_initialized;
}

} // anonymous namespace

template <>
int32_t registerCallback<true>(uint32_t moduleId, ProfCommandHandle handle) {
  if (!isMsprofAvailable() || !g_MsprofRegisterCallback) {
    return -1;  // Not available
  }
  return g_MsprofRegisterCallback(moduleId, handle);
}

template <>
int32_t registerCallback<false>(uint32_t moduleId, ProfCommandHandle handle) {
  if (!isMsprofAvailable() || !g_MsprofRegisterCallback) {
    return -1;
  }
  return g_MsprofRegisterCallback(moduleId, handle);
}

template <>
int32_t reportApi<true>(uint32_t agingFlag, const struct MsprofApi *api) {
  if (!isMsprofAvailable() || !g_MsprofReportApi) {
    return -1;
  }
  return g_MsprofReportApi(agingFlag, api);
}

template <>
int32_t reportApi<false>(uint32_t agingFlag, const struct MsprofApi *api) {
  if (!isMsprofAvailable() || !g_MsprofReportApi) {
    return -1;
  }
  return g_MsprofReportApi(agingFlag, api);
}

template <>
int32_t reportCompactInfo<true>(uint32_t agingFlag, const void *data, uint32_t length) {
  if (!isMsprofAvailable() || !g_MsprofReportCompactInfo) {
    return -1;
  }
  return g_MsprofReportCompactInfo(agingFlag, data, length);
}

template <>
int32_t reportCompactInfo<false>(uint32_t agingFlag, const void *data, uint32_t length) {
  if (!isMsprofAvailable() || !g_MsprofReportCompactInfo) {
    return -1;
  }
  return g_MsprofReportCompactInfo(agingFlag, data, length);
}

template <>
int32_t reportAdditionalInfo<true>(uint32_t agingFlag, const void *data, uint32_t length) {
  if (!isMsprofAvailable() || !g_MsprofReportAdditionalInfo) {
    return -1;
  }
  return g_MsprofReportAdditionalInfo(agingFlag, data, length);
}

template <>
int32_t reportAdditionalInfo<false>(uint32_t agingFlag, const void *data, uint32_t length) {
  if (!isMsprofAvailable() || !g_MsprofReportAdditionalInfo) {
    return -1;
  }
  return g_MsprofReportAdditionalInfo(agingFlag, data, length);
}

template <>
uint64_t getHashId<true>(const char *hashInfo, size_t length) {
  if (!isMsprofAvailable() || !g_MsprofGetHashId) {
    // Return a simple hash if msprof is not available
    uint64_t hash = 0;
    for (size_t i = 0; i < length; ++i) {
      hash = hash * 31 + hashInfo[i];
    }
    return hash;
  }
  return g_MsprofGetHashId(hashInfo, length);
}

template <>
uint64_t getHashId<false>(const char *hashInfo, size_t length) {
  if (!isMsprofAvailable() || !g_MsprofGetHashId) {
    uint64_t hash = 0;
    for (size_t i = 0; i < length; ++i) {
      hash = hash * 31 + hashInfo[i];
    }
    return hash;
  }
  return g_MsprofGetHashId(hashInfo, length);
}

template <>
uint64_t getSysCycleTime<true>() {
  if (!isMsprofAvailable() || !g_MsprofSysCycleTime) {
    // Fallback to std::chrono
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  return g_MsprofSysCycleTime();
}

template <>
uint64_t getSysCycleTime<false>() {
  if (!isMsprofAvailable() || !g_MsprofSysCycleTime) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  return g_MsprofSysCycleTime();
}

template <>
int32_t init<true>(uint32_t dataType, void *data, uint32_t dataLen) {
  if (!isMsprofAvailable() || !g_MsprofInit) {
    return -1;
  }
  return g_MsprofInit(dataType, data, dataLen);
}

template <>
int32_t init<false>(uint32_t dataType, void *data, uint32_t dataLen) {
  if (!isMsprofAvailable() || !g_MsprofInit) {
    return -1;
  }
  return g_MsprofInit(dataType, data, dataLen);
}

template <>
int32_t finalize<true>() {
  // msprof doesn't have an explicit finalize, but we can clean up our handle
  if (g_msprofHandle) {
    // Don't actually close it as it might still be in use
    // dlclose(g_msprofHandle);
    // g_msprofHandle = nullptr;
  }
  return 0;
}

template <>
int32_t finalize<false>() {
  return 0;
}

} // namespace msprof
} // namespace proton
