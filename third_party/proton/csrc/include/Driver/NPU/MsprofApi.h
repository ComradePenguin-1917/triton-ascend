#ifndef PROTON_DRIVER_NPU_MSPROF_API_H_
#define PROTON_DRIVER_NPU_MSPROF_API_H_

#include "Driver/NPU/AscendApi.h"
#include <cstdint>

// Forward declare msprof structures and types
extern "C" {

// msprof common types
typedef void* VOID_PTR;
typedef int32_t (*ProfCommandHandle)(uint32_t type, VOID_PTR data, uint32_t len);

// msprof report levels
#define MSPROF_REPORT_NODE_LEVEL 10000U

// msprof report types for node level
#define MSPROF_REPORT_NODE_LAUNCH_TYPE 5U
#define MSPROF_REPORT_NODE_BASIC_INFO_TYPE 0U
#define MSPROF_REPORT_NODE_TENSOR_INFO_TYPE 1U

// msprof magic number
#define MSPROF_REPORT_DATA_MAGIC_NUM 0x5A5AU

// GE task types
#define MSPROF_GE_TASK_TYPE_AI_CORE 0U
#define MSPROF_GE_TASK_TYPE_AIV 1U

// Tensor kinds
#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

// Tensor data limits
#define MSPROF_GE_TENSOR_DATA_NUM 5
#define MSPROF_GE_TENSOR_DATA_SHAPE_LEN 8

// Tensor types
#define MSPROF_GE_TENSOR_TYPE_INPUT 0
#define MSPROF_GE_TENSOR_TYPE_OUTPUT 1

// msprof API structure for reporting launch timestamps
struct MsprofApi {
    uint16_t magicNumber;  // Must be MSPROF_REPORT_DATA_MAGIC_NUM (0x5A5A)
    uint16_t level;        // MSPROF_REPORT_NODE_LEVEL
    uint32_t type;         // MSPROF_REPORT_NODE_LAUNCH_TYPE
    uint32_t threadId;
    uint32_t reserve;
    uint64_t beginTime;
    uint64_t endTime;
    uint64_t itemId;  // hash ID of kernel name
};

// msprof compact info structure for basic operator info
struct MsprofGeNodeBasicInfo {
    uint64_t opName;
    uint64_t opType;
    uint32_t taskType;
    uint32_t blockDim;
};

struct MsprofCompactInfo {
    uint16_t magicNumber;  // Must be MSPROF_REPORT_DATA_MAGIC_NUM (0x5A5A)
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint64_t timeStamp;
    union {
        struct MsprofGeNodeBasicInfo nodeBasicInfo;
        uint8_t rawData[128];  // Reserve space for other types
    } data;
};

// msprof tensor info structure
struct MsprofGeTensorData {
    uint32_t tensorType;  // input/output
    uint32_t format;      // data format
    uint32_t dataType;    // data type
    int64_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN];
};

struct MsprofTensorInfo {
    uint64_t opName;
    uint32_t tensorNum;
    struct MsprofGeTensorData tensorData[MSPROF_GE_TENSOR_DATA_NUM];
};

struct MsprofAdditionalInfo {
    uint16_t magicNumber;  // Must be MSPROF_REPORT_DATA_MAGIC_NUM (0x5A5A)
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint64_t timeStamp;
    uint8_t data[512];  // Contains MsprofTensorInfo
};

// Command handle structure for profiling control
#define MSPROF_MAX_DEV_NUM 64
#define PATH_LEN_MAX 1023
#define PARAM_LEN_MAX 4095

struct MsprofCommandHandleParams {
    uint32_t pathLen;
    uint32_t storageLimit;
    uint32_t profDataLen;
    char path[PATH_LEN_MAX + 1];
    char profData[PARAM_LEN_MAX + 1];
};

struct MsprofCommandHandle {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[MSPROF_MAX_DEV_NUM];
    uint32_t modelId;
    uint32_t type;
    uint32_t cacheFlag;
    struct MsprofCommandHandleParams params;
};

} // extern "C"

namespace proton {
namespace msprof {

// Template wrappers for msprof APIs
template <bool CheckErrors = true>
int32_t registerCallback(uint32_t moduleId, ProfCommandHandle handle);

template <bool CheckErrors = true>
int32_t reportApi(uint32_t agingFlag, const struct MsprofApi *api);

template <bool CheckErrors = true>
int32_t reportCompactInfo(uint32_t agingFlag, const void *data, uint32_t length);

template <bool CheckErrors = true>
int32_t reportAdditionalInfo(uint32_t agingFlag, const void *data, uint32_t length);

template <bool CheckErrors = true>
uint64_t getHashId(const char *hashInfo, size_t length);

template <bool CheckErrors = true>
uint64_t getSysCycleTime();

// Initialization and finalization
template <bool CheckErrors = true>
int32_t init(uint32_t dataType, void *data, uint32_t dataLen);

template <bool CheckErrors = true>
int32_t finalize();

} // namespace msprof
} // namespace proton

#endif // PROTON_DRIVER_NPU_MSPROF_API_H_
