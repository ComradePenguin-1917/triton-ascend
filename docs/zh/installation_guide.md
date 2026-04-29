# 安装指南

Proton 插桩依赖 AscendNPU-IR 子模块，Triton-Ascend 须从源码构建。

## 环境准备

### Python版本要求

当前Triton-Ascend要求的Python版本为:**py3.9-py3.11**。

### 安装Ascend CANN

异构计算架构CANN（Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，
向上支持多种AI框架，包括MindSpore、PyTorch、TensorFlow等，向下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台。

您可以访问昇腾社区官网，根据其提供的[社区软件安装指引](https://www.hiascend.com/cann/download)完成 CANN 的安装与配置。开发者选择CANN版本、产品系列、CPU架构、操作系统和安装方式便可找到对应的安装命令。

在安装过程中，CANN 版本"**{version}**"请选择如下版本之一。建议下载安装 8.5.0 版本:
- 注：如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。root用户：`/usr/local/Ascend`，非root用户：`${HOME}/Ascend`，`${HOME}`为当前用户目录。
上述环境变量配置只在当前窗口生效，用户可以按需将```source ${HOME}/Ascend/ascend-toolkit/set_env.sh```命令写入环境变量配置文件（如.bashrc文件）。

**CANN版本：**

- 商用版

| Triton-Ascend版本 | CANN商用版本 | CANN发布日期 |
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2         | 2025/11/20         |
|                   | CANN 8.3.RC1         | 2025/10/30         |

- 社区版

| Triton-Ascend版本 | CANN社区版本 | CANN发布日期 |
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2         | 2025/11/20         |
|                   | CANN 8.5.0.alpha001  | 2025/11/12         |
|                   | CANN 8.3.RC1         | 2025/10/30         |


### 安装torch_npu

当前配套的torch_npu版本为2.7.1版本。

```bash
pip install torch_npu==2.7.1
```

注：如果出现报错`ERROR: No matching distribution found for torch==2.7.1+cpu`，可以尝试手动安装torch后再安装torch_npu。
```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## 源码构建

### 系统要求

- GCC >= 9.4.0
- GLIBC >= 2.27
- CMake >= 3.28
- Ninja >= 1.12.0

### 依赖

#### 安装系统库依赖

安装zlib1g-dev/lld/clang，可选择安装ccache包用于加速构建。

- 推荐版本 clang >= 15
- 推荐版本 lld >= 15

```bash
# Ubuntu:
sudo apt update
sudo apt install zlib1g-dev clang-15 lld-15
sudo apt install ccache  # 可选
```

Triton-Ascend的构建强依赖zlib1g-dev，如果您使用yum源，请参考如下命令安装：

```bash
sudo yum install -y zlib-devel
```

#### 安装python依赖

```bash
pip install ninja cmake wheel pybind11
```

#### LLVM 依赖

Triton-Ascend 本体需要预编译的 LLVM，`bishengir-compile` 则通过子模块自动拉取其所需的 LLVM 版本。两者版本不同，各自独立管理。

**编译 triton-ascend 所需的 LLVM**（commit `b5cc222d`）：

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

编译完成后，配置环境变量：

```bash
export LLVM_SYSPATH=/path/to/llvm-install
```

可将此行写入 `~/.bashrc` 或 `setup_npu_env.sh` 中以持久化。

### 克隆并构建

构建前请确保已激活目标 Python 环境（安装 torch_npu 的环境）。

```bash
source activate your_env_name
export LLVM_SYSPATH=/path/to/llvm-install
git clone https://gitcode.com/Ascend/triton-ascend.git
cd triton-ascend
bash build.sh
```

构建脚本自动完成：
1. 初始化子模块（含支持 Proton 的 AscendNPU-IR）
2. 从源码编译 `bishengir-compile`（首次构建需配合 `--apply-patches` 打补丁）
3. 编译安装 Triton-Ascend（默认启用 Proton）

关键环境变量（可覆盖）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLVM_SYSPATH` | `/shared/llvm/triton-ascend/` | 预编译 LLVM 安装路径 |
| `TRITON_BUILD_PROTON` | `ON` | 启用 Proton profiler |
| `MAX_JOBS` | `64` | 并行编译线程数 |
| `TRITON_BUILD_WITH_CCACHE` | `true` | 使用 ccache 加速重复编译 |

### 运行示例

安装运行时依赖：

```bash
cd triton-ascend && pip install -r requirements.txt
```

运行实例: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 ./third_party/ascend/tutorials/01-vector-add.py
```

观察到类似的输出即说明环境配置正确。

```bash
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```
