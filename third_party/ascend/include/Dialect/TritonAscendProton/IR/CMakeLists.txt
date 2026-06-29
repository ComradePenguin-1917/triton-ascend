# Generate dialect definitions
set(LLVM_TARGET_DEFINITIONS TritonAscendProtonDialect.td)
mlir_tablegen(TritonAscendProtonDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TritonAscendProtonDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(TritonAscendProtonDialectIncGen)

# Generate attribute definitions from TableGen
set(LLVM_TARGET_DEFINITIONS TritonAscendProtonAttrDefs.td)
mlir_tablegen(TritonAscendProtonAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonAscendProtonAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(TritonAscendProtonEnums.h.inc -gen-enum-decls)
mlir_tablegen(TritonAscendProtonEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TritonAscendProtonAttrDefsIncGen)

# Generate operation definitions from TableGen
set(LLVM_TARGET_DEFINITIONS TritonAscendProtonOps.td)
mlir_tablegen(TritonAscendProtonOps.h.inc -gen-op-decls)
mlir_tablegen(TritonAscendProtonOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TritonAscendProtonOpsIncGen)

