#!/usr/bin/env bash
# Triton-Ascend build script with Proton profiler support.
#
# Usage: ./build.sh [clean] [debug|relwithdebinfo]
#
# Prerequisites:
#   - Active Python environment with torch, torch_npu, ninja, cmake, pybind11
#   - LLVM_SYSPATH set to LLVM installation path (commit fad3272, see docs)
#
# Environment variables (optional):
#   TRITON_BUILD_PROTON           - enable Proton profiler (default: ON)
#   TRITON_BUILD_WITH_CCACHE      - use ccache (default: true)
#   TRITON_BUILD_WITH_CLANG_LLD   - use clang + lld (default: true)
#   MAX_JOBS                      - parallel build jobs (default: 64)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE="debug"
DO_CLEAN=0
for arg in "$@"; do
  case "$arg" in
    clean)       DO_CLEAN=1 ;;
    debug)       MODE="debug" ;;
    relwithdebinfo) MODE="relwithdebinfo" ;;
    *)
      echo "Usage: ./build.sh [clean] [debug|relwithdebinfo]" >&2
      exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Activate environment
# ---------------------------------------------------------------------------
source "${REPO_ROOT}/setup_npu_env.sh"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
if [[ ${DO_CLEAN} -eq 1 ]]; then
  echo "=== Cleaning build artifacts ==="
  rm -rf "${REPO_ROOT}/build" "${REPO_ROOT}/dist" "${REPO_ROOT}/triton.egg-info"
fi

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
if [ -z "${LLVM_SYSPATH:-}" ]; then
  echo "Error: LLVM_SYSPATH is not set." >&2
  echo "Please set it to your LLVM installation path, e.g.:" >&2
  echo "  export LLVM_SYSPATH=/path/to/llvm-install" >&2
  echo "See docs/zh/installation_guide.md for LLVM build instructions." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------
unset TRITON_PLUGIN_DIRS
export LLVM_SYSPATH
export TRITON_BUILD_WITH_CCACHE="${TRITON_BUILD_WITH_CCACHE:-true}"
export TRITON_BUILD_WITH_CLANG_LLD="${TRITON_BUILD_WITH_CLANG_LLD:-true}"
export TRITON_BUILD_PROTON="${TRITON_BUILD_PROTON:-ON}"
export TRITON_WHEEL_NAME="${TRITON_WHEEL_NAME:-triton-ascend}"
export TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:--DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DTRITON_BUILD_UT=OFF -DASCEND_TOOLKIT_PATH:PATH=${ASCEND_HOME_PATH}}"
export MAX_JOBS="${MAX_JOBS:-64}"

unset DEBUG REL_WITH_DEB_INFO TRITON_BUILD_DEBUG TRITON_BUILD_RELWITHDEBINFO
case "${MODE}" in
  debug)          export DEBUG=1 TRITON_BUILD_DEBUG=ON ;;
  relwithdebinfo) export REL_WITH_DEB_INFO=1 TRITON_BUILD_RELWITHDEBINFO=ON ;;
esac

# ---------------------------------------------------------------------------
# Print build info
# ---------------------------------------------------------------------------
echo "=== Triton-Ascend Build Script ==="
echo "Repository      : ${REPO_ROOT}"
echo "Build mode      : ${MODE}"
echo "LLVM_SYSPATH    : ${LLVM_SYSPATH}"
echo "ASCEND_HOME     : ${ASCEND_TOOLKIT_HOME:-<unset>}"
echo "MAX_JOBS        : ${MAX_JOBS}"
echo "TRITON_PROTON   : ${TRITON_BUILD_PROTON}"
echo "Compiler        : clang/lld via setup.py"
echo "bishengir       : $(command -v bishengir-compile 2>/dev/null || echo not-found)"
echo

# ---------------------------------------------------------------------------
# Step 1: Update submodules
# ---------------------------------------------------------------------------
echo "=== Updating submodules ==="
git submodule update --init --recursive
(cd "${REPO_ROOT}/third_party/ascend/AscendNPU-IR" && git fetch origin master && git reset --hard origin/master)

# ---------------------------------------------------------------------------
# Step 2: Build bishengir-compile (Proton op lowering compiler)
# ---------------------------------------------------------------------------
BISHENGIR_SRC="${REPO_ROOT}/third_party/ascend/AscendNPU-IR"
BISHENGIR_BUILD="${BISHENGIR_SRC}/build"
if [ ! -f "${BISHENGIR_BUILD}/bin/bishengir-compile" ]; then
    echo "=== Building bishengir-compile ==="
    (cd "${BISHENGIR_SRC}" && bash build-tools/apply_patches.sh 2>/dev/null; true)
    LLVM_SRC="${BISHENGIR_SRC}/third-party/llvm-project/llvm"
    cmake -S "${LLVM_SRC}" -B "${BISHENGIR_BUILD}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_EXTERNAL_PROJECTS=bishengir \
        -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="${BISHENGIR_SRC}" \
        -DLLVM_TARGETS_TO_BUILD=AArch64 \
        -DBSPUB_DAVINCI_BISHENGIR=ON \
        -DBISHENGIR_PUBLISH=ON \
        -DBISHENGIR_BUILD_TEMPLATE=ON \
        -DBISHENG_COMPILER_PATH="${ASCEND_HOME_PATH}/bin" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build "${BISHENGIR_BUILD}" --target bishengir-compile -j "${MAX_JOBS}"
fi
if [ -f "${BISHENGIR_BUILD}/bin/bishengir-compile" ]; then
    cp "${BISHENGIR_BUILD}/bin/bishengir-compile" \
       "${ASCEND_HOME_PATH}/tools/bishengir/bin/bishengir-compile"
    echo "  installed to CANN"
fi

# ---------------------------------------------------------------------------
# Step 3: Build and install Triton-Ascend
# ---------------------------------------------------------------------------
echo "=== Building Triton-Ascend ==="
cd "${REPO_ROOT}"
python3 setup.py install

echo
echo "Build completed successfully."
