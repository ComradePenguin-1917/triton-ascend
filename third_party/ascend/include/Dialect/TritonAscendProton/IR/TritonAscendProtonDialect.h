//===----------------------------------------------------------------------===//
// Triton Ascend Proton Dialect Header
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_PROTON_DIALECT_H
#define TRITON_ASCEND_PROTON_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// Include generated dialect declarations
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonDialect.h.inc"

// Include generated enums and attributes
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonEnums.h.inc"

namespace mlir {
namespace triton {
namespace proton {

// Initialize the Triton Ascend Proton dialect
void registerTritonAscendProtonDialect(DialectRegistry &registry);

} // namespace proton
} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_PROTON_DIALECT_H
