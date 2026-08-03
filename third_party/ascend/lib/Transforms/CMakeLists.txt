add_mlir_library(TritonAscendProtonTransforms
  ProtonToHIVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Dialect/TritonAscendProton
  ${CMAKE_CURRENT_BINARY_DIR}/../../include/Dialect/TritonAscendProton
  ${ASCENDNPU_IR_SRC_DIR}/bishengir/include
  ${ASCENDNPU_IR_BINARY_DIR}/bishengir/include

  DEPENDS
  TritonAscendProtonDialectIncGen
  TritonAscendProtonAttrDefsIncGen
  TritonAscendProtonOpsIncGen

  LINK_LIBS PUBLIC
  TritonAscendProtonDialect
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
)

# Add include directories for generated files
target_include_directories(TritonAscendProtonTransforms PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../include
  ${ASCENDNPU_IR_SRC_DIR}/bishengir/include
  ${ASCENDNPU_IR_BINARY_DIR}/bishengir/include
)
