//===----------------------------------------------------------------------===//
// Triton Ascend Proton Transformation Passes
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_PROTON_TRANSFORMS_PASSES_H
#define TRITON_ASCEND_PROTON_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
class MLIRContext;
class RewritePatternSet;

namespace triton {
namespace proton {

//===----------------------------------------------------------------------===//
// Pass to lower Triton Ascend Proton ops to HIVM
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> createTritonAscendProtonToHIVMPass();

//===----------------------------------------------------------------------===//
// Pass to lower ReadCycleCounterOp to GetSysCntOp (must run after CSE)
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> createTritonAscendProtonLowerCycleCounterPass();

//===----------------------------------------------------------------------===//
// Pass to remove Proton instrumentation ops
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> createRemoveTritonAscendProtonOpsPass();

//===----------------------------------------------------------------------===//
// Pass to optimize Proton instrumentation placement
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> createOptimizeTritonAscendProtonPlacementPass();

//===----------------------------------------------------------------------===//
// Pattern population functions
//===----------------------------------------------------------------------===//
void populateTritonAscendProtonToHIVMPatterns(RewritePatternSet &patterns);
void populateTritonAscendProtonRemovalPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Register all Triton Ascend Proton transformation passes.
void registerTritonAscendProtonPasses();

/// Register just the ProtonToHIVM pass for use in triton-adapter-opt
inline void registerProtonToHIVMPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createTritonAscendProtonToHIVMPass();
  });
}

} // namespace proton
} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_PROTON_TRANSFORMS_PASSES_H

