//===----------------------------------------------------------------------===//
// Triton Ascend Proton Operations Header
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_PROTON_OPS_H
#define TRITON_ASCEND_PROTON_OPS_H

#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonOps.h.inc"

#endif // TRITON_ASCEND_PROTON_OPS_H
