//===----------------------------------------------------------------------===//
// Triton Ascend Proton Operations Definition
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_PROTON_OPS
#define TRITON_ASCEND_PROTON_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "TritonAscendProtonDialect.td"
include "TritonAscendProtonAttrDefs.td"

//===----------------------------------------------------------------------===//
// Record Operation - Compatible with original Proton
//===----------------------------------------------------------------------===//

def TritonAscendProton_RecordOp : TritonAscendProton_Op<"record", []> {
  let summary = "Record an event for Ascend NPU profiling";
  
  let description = [{
    Records profiling events for Ascend NPU execution. This operation
    annotates regions of IR where performance metrics should be collected.
    
    Supports:
    - AICore utilization tracking
    - AICPU overhead measurement
    - Memory bandwidth monitoring
    - Synchronization overhead
    
    Example:
    ```mlir
    ascend_proton.record start "kernel"
    // ... kernel operations ...
    ascend_proton.record end "kernel"
    ```
  }];
  
  let arguments = (ins
    UnitAttr:$isStart,
    StrAttr:$name,
    OptionalAttr<I32Attr>:$aicoreId,
    OptionalAttr<I32Attr>:$streamId
  );
  
  let assemblyFormat = [{
    (`start` $isStart^):(`end`)? $name 
    (`aicore` `=` $aicoreId^)?
    (`stream` `=` $streamId^)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Metric Collection Operation - NPU specific
//===----------------------------------------------------------------------===//

def TritonAscendProton_MetricOp : TritonAscendProton_Op<"metric", []> {
  let summary = "Collect NPU-specific metrics";
  
  let description = [{
    Collects hardware performance metrics specific to Ascend NPU.
    This includes AICore utilization, memory bandwidth, and synchronization overhead.
  }];
  
  let arguments = (ins
    TritonAscendProton_MetricTypeAttr:$metricType,
    StrAttr:$name
  );
  
  let results = (outs I64:$value);
  
  let assemblyFormat = [{
    $metricType $name attr-dict `:` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// Barrier Tracking Operation
//===----------------------------------------------------------------------===//

def TritonAscendProton_BarrierOp : TritonAscendProton_Op<"barrier", []> {
  let summary = "Track NPU synchronization barriers";
  
  let description = [{
    Marks synchronization barriers in the NPU execution flow.
    Used to measure synchronization overhead and identify bottlenecks.
  }];
  
  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<I32Attr>:$barrierType
  );
  
  let assemblyFormat = [{
    $name (`type` `=` $barrierType^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Workspace Memory Tracking
//===----------------------------------------------------------------------===//

def TritonAscendProton_WorkspaceOp : TritonAscendProton_Op<"workspace", []> {
  let summary = "Track workspace memory allocation";
  
  let description = [{
    Tracks workspace memory allocation and usage patterns on Ascend NPU.
    Helps optimize memory allocation strategies.
  }];
  
  let arguments = (ins
    I64Attr:$size,
    StrAttr:$name,
    OptionalAttr<I32Attr>:$memoryType
  );
  
  let assemblyFormat = [{
    $name `size` `=` $size (`type` `=` $memoryType^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Multi-buffer Tracking
//===----------------------------------------------------------------------===//

def TritonAscendProton_MultibufferOp : TritonAscendProton_Op<"multibuffer", []> {
  let summary = "Track multi-buffering configuration";
  
  let description = [{
    Records multi-buffering configuration for optimizing data movement
    between different memory hierarchies on Ascend NPU.
  }];
  
  let arguments = (ins
    I32Attr:$bufferCount,
    StrAttr:$name,
    OptionalAttr<I64Attr>:$bufferSize
  );
  
  let assemblyFormat = [{
    $name `count` `=` $bufferCount (`size` `=` $bufferSize^)? attr-dict
  }];
}

#endif // TRITON_ASCEND_PROTON_OPS
