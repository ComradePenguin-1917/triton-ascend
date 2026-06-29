#ifndef TRITON_ASCEND_PROTON_ATTR_DEFS
#define TRITON_ASCEND_PROTON_ATTR_DEFS

include "mlir/IR/EnumAttr.td"
include "TritonAscendProtonDialect.td"

def TritonAscendProton_Cycle : I32EnumAttrCase<"Cycle", 0, "cycle">;
def TritonAscendProton_AICore : I32EnumAttrCase<"AICore", 1, "aicore">;
def TritonAscendProton_AICPU : I32EnumAttrCase<"AICPU", 2, "aicpu">;
def TritonAscendProton_Memory : I32EnumAttrCase<"Memory", 3, "memory">;
def TritonAscendProton_Sync : I32EnumAttrCase<"Sync", 4, "sync">;

def TritonAscendProton_MetricTypeAttr : I32EnumAttr<
    "MetricType", "Ascend NPU performance metric types",
    [TritonAscendProton_Cycle, TritonAscendProton_AICore, TritonAscendProton_AICPU,
     TritonAscendProton_Memory, TritonAscendProton_Sync]> {
  let cppNamespace = "::mlir::triton::proton";
}

def TritonAscendProton_None : I32EnumAttrCase<"None", 0, "none">;
def TritonAscendProton_Selective : I32EnumAttrCase<"Selective", 1, "selective">;
def TritonAscendProton_Full : I32EnumAttrCase<"Full", 2, "full">;

def TritonAscendProton_SamplingStrategyAttr : I32EnumAttr<
    "SamplingStrategy", "Profiling sampling strategy",
    [TritonAscendProton_None, TritonAscendProton_Selective, TritonAscendProton_Full]> {
  let cppNamespace = "::mlir::triton::proton";
}

def TritonAscendProton_AICoreLevelGranularity : I32EnumAttrCase<"AICore", 0, "aicore">;
def TritonAscendProton_BlockGranularity : I32EnumAttrCase<"Block", 1, "block">;
def TritonAscendProton_KernelGranularity : I32EnumAttrCase<"Kernel", 2, "kernel">;

def TritonAscendProton_GranularityAttr : I32EnumAttr<
    "Granularity", "Profiling granularity level",
    [TritonAscendProton_AICoreLevelGranularity, TritonAscendProton_BlockGranularity,
     TritonAscendProton_KernelGranularity]> {
  let cppNamespace = "::mlir::triton::proton";
}

def TritonAscendProton_Circular : I32EnumAttrCase<"Circular", 0, "circular">;
def TritonAscendProton_Flush : I32EnumAttrCase<"Flush", 1, "flush">;

def TritonAscendProton_BufferStrategyAttr : I32EnumAttr<
    "BufferStrategy", "Profiling buffer management strategy",
    [TritonAscendProton_Circular, TritonAscendProton_Flush]> {
  let cppNamespace = "::mlir::triton::proton";
}

def TritonAscendProton_GlobalMemory : I32EnumAttrCase<"GlobalMemory", 0, "global">;
def TritonAscendProton_LocalMemory : I32EnumAttrCase<"LocalMemory", 1, "local">;
def TritonAscendProton_UnifiedBuffer : I32EnumAttrCase<"UnifiedBuffer", 2, "ub">;
def TritonAscendProton_L1Buffer : I32EnumAttrCase<"L1Buffer", 3, "l1">;

def TritonAscendProton_BufferTypeAttr : I32EnumAttr<
    "BufferType", "Ascend NPU memory hierarchy types",
    [TritonAscendProton_GlobalMemory, TritonAscendProton_LocalMemory,
     TritonAscendProton_UnifiedBuffer, TritonAscendProton_L1Buffer]> {
  let cppNamespace = "::mlir::triton::proton";
}

#endif // TRITON_ASCEND_PROTON_ATTR_DEFS
