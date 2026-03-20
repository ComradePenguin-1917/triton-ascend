add_mlir_dialect_library(TritonAscendProtonDialect
  TritonAscendProtonDialect.cpp
  TritonAscendProtonOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/Dialect/TritonAscendProton/IR
  ${CMAKE_CURRENT_BINARY_DIR}/../../../../include/Dialect/TritonAscendProton/IR

  DEPENDS
  TritonAscendProtonDialectIncGen
  TritonAscendProtonAttrDefsIncGen
  TritonAscendProtonOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

# Add include directories for generated files
target_include_directories(TritonAscendProtonDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../../include
)
