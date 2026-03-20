#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="${REPO_ROOT}/python"

CONDA_ACTIVATE="/home/lijihang/program/miniforge3/bin/activate"
CONDA_ENV_NAME="py310"
LLVM_SYSPATH_DEFAULT="/home/lijihang/program/llvm/b5cc222d/"

MODE="debug"
DO_CLEAN=0

for arg in "$@"; do
  case "$arg" in
    clean)
      DO_CLEAN=1
      ;;
    debug)
      MODE="debug"
      ;;
    relwithdebinfo)
      MODE="relwithdebinfo"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: ./build.sh [clean] [debug|relwithdebinfo]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONDA_ACTIVATE}" ]]; then
  echo "Conda activate script not found: ${CONDA_ACTIVATE}" >&2
  exit 1
fi

source "${CONDA_ACTIVATE}" "${CONDA_ENV_NAME}"

source "${REPO_ROOT}/setup_npu_env.sh"

if [[ ${DO_CLEAN} -eq 1 ]]; then
  echo "Cleaning build artifacts..."
  rm -rf "${PYTHON_DIR}/build" \
         "${PYTHON_DIR}/dist" \
         "${PYTHON_DIR}/triton.egg-info" \
         "${REPO_ROOT}/build"
fi

export LLVM_SYSPATH="${LLVM_SYSPATH:-${LLVM_SYSPATH_DEFAULT}}"
unset TRITON_PLUGIN_DIRS
export TRITON_BUILD_WITH_CCACHE="${TRITON_BUILD_WITH_CCACHE:-true}"
export TRITON_BUILD_WITH_CLANG_LLD="${TRITON_BUILD_WITH_CLANG_LLD:-true}"
export TRITON_BUILD_PROTON="${TRITON_BUILD_PROTON:-ON}"
export TRITON_WHEEL_NAME="${TRITON_WHEEL_NAME:-triton-ascend}"
export TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:--DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DTRITON_BUILD_UT=OFF -DASCEND_TOOLKIT_PATH:PATH=${ASCEND_HOME_PATH}}"
export MAX_JOBS="${MAX_JOBS:-64}"

unset DEBUG
unset REL_WITH_DEB_INFO
unset TRITON_BUILD_DEBUG
unset TRITON_BUILD_RELWITHDEBINFO

case "${MODE}" in
  debug)
    export DEBUG=1
    export TRITON_BUILD_DEBUG=ON
    ;;
  relwithdebinfo)
    export REL_WITH_DEB_INFO=1
    export TRITON_BUILD_RELWITHDEBINFO=ON
    ;;
esac

echo "=== Triton-Ascend Build Script ==="
echo "Repository      : ${REPO_ROOT}"
echo "Python dir      : ${PYTHON_DIR}"
echo "Conda env       : ${CONDA_ENV_NAME}"
echo "Build mode      : ${MODE}"
echo "LLVM_SYSPATH    : ${LLVM_SYSPATH}"
echo "ASCEND_HOME     : ${ASCEND_HOME_PATH:-<unset>}"
echo "NPU env script  : ${REPO_ROOT}/setup_npu_env.sh"
# echo "Plugin dir      : ${TRITON_PLUGIN_DIRS}"
echo "MAX_JOBS        : ${MAX_JOBS}"
echo "TRITON_PROTON   : ${TRITON_BUILD_PROTON}"
echo "Compiler        : clang/lld via setup.py"
echo "bishengir       : $(command -v bishengir-compile 2>/dev/null || echo not-found)"
echo

cd "${PYTHON_DIR}"
python3 setup.py install

echo
echo "Build completed successfully."
