//===----------------------------------------------------------------------===//
// Triton Ascend Proton to HIVM Conversion
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_PROTON_CONVERSION_PROTON_TO_HIVM_H
#define TRITON_ASCEND_PROTON_CONVERSION_PROTON_TO_HIVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {
namespace proton {

// Register the Proton to HIVM conversion pass
void registerConvertProtonToHIVMPass();

// Create the Proton to HIVM conversion pass
std::unique_ptr<Pass> createConvertProtonToHIVMPass();

} // namespace proton
} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_PROTON_CONVERSION_PROTON_TO_HIVM_H
