#!/bin/bash
# Ascend NPU Environment Setup
#
# This script sets up environment variables required by Triton-Ascend.
# It should be sourced:  source setup_npu_env.sh

echo "=== Setting up Ascend NPU Environment ==="

# ---------------------------------------------------------------------------
# Sanitize variables that CANN's set_env.sh expects (avoids unbound errors)
# ---------------------------------------------------------------------------
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

# ---------------------------------------------------------------------------
# Source CANN environment
# ---------------------------------------------------------------------------
CANN_SET_ENV=""
for candidate in ~/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
  if [[ -f "${candidate}" ]]; then
    CANN_SET_ENV="${candidate}"
    break
  fi
done
if [[ -z "${CANN_SET_ENV}" ]]; then
  echo "ERROR: CANN set_env.sh not found" >&2
  exit 1
fi
source "${CANN_SET_ENV}"

# ---------------------------------------------------------------------------
# Locate workspace root
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# LLVM library path
# ---------------------------------------------------------------------------
export LD_LIBRARY_PATH="${LLVM_SYSPATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ---------------------------------------------------------------------------
# BiSheng compiler tools (clang, lld)
# ---------------------------------------------------------------------------
export PATH="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin:${PATH}"

# ---------------------------------------------------------------------------
# BiShengIR compiler for Proton lowering (bishengir-compile)
# ---------------------------------------------------------------------------
export PATH="${ASCEND_HOME_PATH}/tools/bishengir/bin:${PATH}"
export TRITON_NPU_COMPILER_PATH="${ASCEND_HOME_PATH}/tools/bishengir/bin"

# ---------------------------------------------------------------------------
# Additional Ascend runtime paths
# ---------------------------------------------------------------------------
export TBE_IMPL_PATH="${ASCEND_TOOLKIT_HOME}/lib64"
export CANN_PATH="${ASCEND_TOOLKIT_HOME}"
export INSTALL_DIR="${ASCEND_TOOLKIT_HOME}"

# ---------------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------------
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "✓ Environment variables set"
echo
echo "Key paths:"
echo "  ASCEND_HOME_PATH:         ${ASCEND_HOME_PATH}"
echo "  TRITON_NPU_COMPILER_PATH: ${TRITON_NPU_COMPILER_PATH}"
echo "  LLVM lib:                 ${LLVM_SYSPATH}/lib"
echo
echo "Verifying NPU compiler availability..."
if command -v bishengir-compile &> /dev/null; then
  echo "  ✓ bishengir-compile found: $(command -v bishengir-compile)"
else
  echo "  ✗ WARNING: bishengir-compile not found in PATH"
fi
echo
echo "Verifying build tools availability..."
for tool in clang ld.lld; do
  if command -v ${tool} &> /dev/null; then
    echo "  ✓ ${tool} found: $(command -v ${tool})"
  else
    echo "  ✗ WARNING: ${tool} not found in PATH"
  fi
done
