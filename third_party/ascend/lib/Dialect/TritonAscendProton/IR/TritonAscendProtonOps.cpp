//===----------------------------------------------------------------------===//
// Triton Ascend Proton Operations Implementation
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonOps.h"
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::triton::proton;

// Include TableGen generated operation definitions
#define GET_OP_CLASSES
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonOps.cpp.inc"
