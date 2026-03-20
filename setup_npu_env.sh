#!/bin/bash
# Ascend NPU Environment Setup Script

echo "=== Setting up Ascend NPU Environment ==="

# Ascend Toolkit base path
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest/
export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

# Driver libraries (required by CANN)
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# Runtime libraries - following official set_env.sh order
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$(arch):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/lijihang/program/llvm/b5cc222d/lib:$LD_LIBRARY_PATH

# Python packages - must include TBE for operator compilation
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH

# TBE and OPP
export TBE_IMPL_PATH=${ASCEND_TOOLKIT_HOME}/lib64
export ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
export ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}

# Toolchain
export TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
export PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH
# Add bishengir compiler first (preferred over ccec for MLIR compilation)
export PATH=${ASCEND_TOOLKIT_HOME}/bisheng_toolkit/bishengir/bin:$PATH

# Triton NPU Compiler Path
# This is required when bishengir-compile is not in PATH
# Points to the directory containing npuc/bishengir-compile binary
export TRITON_NPU_COMPILER_PATH=${ASCEND_TOOLKIT_HOME}/bisheng_toolkit/bishengir/bin

# CANN specific
export CANN_PATH=${ASCEND_TOOLKIT_HOME}
export INSTALL_DIR=${ASCEND_TOOLKIT_HOME}

# Logging
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# Device
export ASCEND_DEVICE_ID=0

echo "✓ Environment variables set"
echo ""
echo "Key paths:"
echo "  ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
echo "  TRITON_NPU_COMPILER_PATH: $TRITON_NPU_COMPILER_PATH"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:0:100}..."
echo "  PYTHONPATH: ${PYTHONPATH:0:100}..."
echo ""
echo "Verifying NPU compiler availability..."
if command -v bishengir-compile &> /dev/null; then
    echo "  ✓ bishengir-compile found in PATH: $(which bishengir-compile)"
elif [ -f "${TRITON_NPU_COMPILER_PATH}/bishengir-compile" ]; then
    echo "  ✓ bishengir-compile found at: ${TRITON_NPU_COMPILER_PATH}/bishengir-compile"
elif [ -f "${TRITON_NPU_COMPILER_PATH}/npuc" ]; then
    echo "  ✓ npuc found at: ${TRITON_NPU_COMPILER_PATH}/npuc"
else
    echo "  ✗ WARNING: NPU compiler not found!"
    echo "    Expected location: ${TRITON_NPU_COMPILER_PATH}/bishengir-compile"
fi
echo ""
echo "To use this environment:"
echo "  source $(readlink -f $0)"
echo "  python your_script.py"
