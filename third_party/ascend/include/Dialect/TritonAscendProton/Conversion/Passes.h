//===----------------------------------------------------------------------===//
// Triton Ascend Proton Conversion Passes
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_PROTON_CONVERSION_PASSES_H
#define TRITON_ASCEND_PROTON_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {
namespace proton {

/// Register all conversion passes
void registerConversionPasses();

} // namespace proton
} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_PROTON_CONVERSION_PASSES_H

