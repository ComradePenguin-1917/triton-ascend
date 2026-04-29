# Installation Guide

Proton-enabled Triton-Ascend must be built from source.

## Preparing the Environment

### Python Version Requirements

Triton-Ascend requires Python 3.9 to 3.11.

### Installing Ascend CANN

Compute Architecture for Neural Networks (CANN) is a heterogeneous compute architecture developed by Ascend for AI scenarios.
It plays a pivotal bridging role: providing upward integration with multiple AI frameworks (including MindSpore, PyTorch, and TensorFlow), while offering downward support for AI processors and programming. This establishes it as a key platform for improving the computing efficiency of Ascend AI processors.

You can visit the Ascend community website, and install and configure CANN according to the provided [software installation guide](https://www.hiascend.com/cann/download). Developers can select the CANN version, product series, CPU architecture, operating system, and installation method to find the corresponding installation commands.

During the installation, select one of the following CANN versions in *{version}*. It is advisable to download and install version 8.5.0.

- Note: If the installation path is not specified, software will be installed in the default path. The default installation paths are as follows: For the **root** user, the path is `/usr/local/Ascend`. For non-root users, the path is `${HOME}/Ascend`, where `${HOME}` indicates the current user's directory.
The preceding environment variable configurations take effect only in the current window. You can add the `source ${HOME}/Ascend/ascend-toolkit/set_env.sh` command to the environment variable configuration file (such as the .bashrc file) as required.

**CANN version:**

- Commercial edition

| Triton-Ascend Version| CANN Commercial Version| CANN Release Date|
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026-01-16        |
| 3.2.0rc4          | CANN 8.3.RC2         | 2025-11-20        |
|                   | CANN 8.3.RC1         | 2025-10-30        |

- Community edition

| Triton-Ascend Version| CANN Community Version| CANN Release Date|
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026-01-16        |
| 3.2.0rc4          | CANN 8.3.RC2         | 2025-11-20        |
|                   | CANN 8.5.0.alpha001  | 2025-11-12        |
|                   | CANN 8.3.RC1         | 2025-10-30        |


### Installing torch_npu

The current torch_npu version is 2.7.1.

```bash
pip install torch_npu==2.7.1
```

Note: If `ERROR: No matching distribution found for torch==2.7.1+cpu` is displayed, you can manually install Torch and then install torch_npu.
```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Building from Source

### System Requirements

- GCC >= 9.4.0
- GLIBC >= 2.27
- CMake >= 3.28
- Ninja >= 1.12.0

### Dependencies

#### System Library Dependencies

Install zlib1g-dev, LLD and Clang. You can also install ccache to accelerate the build process.

- Recommended version: Clang >= 15
- Recommended version: LLD >= 15

```bash
# Ubuntu:
sudo apt update
sudo apt install zlib1g-dev clang-15 lld-15
sudo apt install ccache # optional
```

Triton-Ascend depends heavily on zlib1g-dev. If you use the yum source, run the following installation command:

```bash
sudo yum install -y zlib-devel
```

#### Python Dependencies

```bash
pip install ninja cmake wheel pybind11
```

#### LLVM Dependency

Triton-Ascend requires a pre-built LLVM, while `bishengir-compile` automatically pulls its own LLVM version via the submodule. The two LLVM versions are independent.

**Build LLVM for triton-ascend** (commit `b5cc222d`):

```bash
git clone --no-checkout https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
mkdir build && cd build
cmake ../llvm -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-15 \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DCMAKE_INSTALL_PREFIX=/path/to/llvm-install \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="lld;llvm;mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
ninja -j$(nproc)
ninja install
```

After building, set the environment variable:

```bash
export LLVM_SYSPATH=/path/to/llvm-install
```

Add this to `~/.bashrc` or `setup_npu_env.sh` to persist it.

### Clone and Build

Ensure the target Python environment (with torch_npu installed) is active before building.

```bash
source activate your_env_name
export LLVM_SYSPATH=/path/to/llvm-install
git clone https://gitcode.com/Ascend/triton-ascend.git
cd triton-ascend
bash build.sh
```

The build script automatically:
1. Initializes submodules (including AscendNPU-IR with Proton support)
2. Builds `bishengir-compile` from source (requires `--apply-patches` on first build)
3. Builds and installs Triton-Ascend with Proton enabled

Key environment variables (overridable):

| Variable | Default | Description |
|---|---|---|
| `LLVM_SYSPATH` | `/shared/llvm/triton-ascend/` | Pre-built LLVM installation path |
| `TRITON_BUILD_PROTON` | `ON` | Enable Proton profiler |
| `MAX_JOBS` | `64` | Parallel build jobs |
| `TRITON_BUILD_WITH_CCACHE` | `true` | Use ccache to accelerate rebuilds |

### Running the Example

Install runtime dependencies:

```bash
cd triton-ascend && pip install -r requirements_dev.txt
```

Run the [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py) example:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 ./third_party/ascend/tutorials/01-vector-add.py
```

Expected output:

```
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```
