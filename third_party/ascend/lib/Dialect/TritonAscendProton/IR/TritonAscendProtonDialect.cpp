//===----------------------------------------------------------------------===//
// Triton Ascend Proton Dialect Implementation
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonDialect.h"
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::proton;

// Include generated dialect definitions
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonDialect.cpp.inc"

// Include generated enum definitions
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void TritonAscendProtonDialect::initialize() {
  // Register operations manually
  addOperations<
      RecordOp,
      MetricOp,
      BarrierOp,
      MultibufferOp,
      ReadCycleCounterOp
      >();
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void mlir::triton::proton::registerTritonAscendProtonDialect(DialectRegistry &registry) {
  registry.insert<TritonAscendProtonDialect>();
}
