//===----------------------------------------------------------------------===//
// Triton Ascend Proton Dialect Definition
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_PROTON_DIALECT
#define TRITON_ASCEND_PROTON_DIALECT

include "mlir/IR/OpBase.td"

def TritonAscendProton_Dialect : Dialect {
  let name = "ascend_proton";
  let summary = "Triton Ascend Proton dialect for NPU profiling and instrumentation";
  let description = [{
    The Triton Ascend Proton dialect provides profiling and instrumentation
    capabilities for Ascend NPU kernels. It supports:
    - Hierarchical profiling scopes
    - Hardware performance counter collection
    - NPU-specific metrics (AICore, AICPU utilization)
    - Integration with Ascend profiling infrastructure
  }];
  
  let cppNamespace = "::mlir::triton::proton";
  
  // Note: We don't use custom attributes or types yet, so we don't need
  // the default attribute/type printer/parser
  // let useDefaultAttributePrinterParser = 1;
  // let useDefaultTypePrinterParser = 1;
  
  let dependentDialects = [];
}

//===----------------------------------------------------------------------===//
// Base class for Triton Ascend Proton dialect ops
//===----------------------------------------------------------------------===//

class TritonAscendProton_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonAscendProton_Dialect, mnemonic, traits>;

#endif // TRITON_ASCEND_PROTON_DIALECT
