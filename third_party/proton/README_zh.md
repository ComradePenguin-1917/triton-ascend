# Proton - Triton 性能分析器

## 简介

Proton 是 Triton 的轻量级性能分析器，用于分析 Python 代码中调用的 GPU/NPU 内核。Proton 提供程序上下文、元数据以及底层 GPU/NPU 内核的硬件性能指标信息。

## 安装

Proton 随 Triton-Ascend 源码构建。请参考[安装指南](../../docs/zh/installation_guide.md)从源码构建，Proton 默认启用。

## 使用方法

### 基本用法

更多示例见 [tutorials](tutorials) 目录。

Proton 可用于分析 Python 代码中的*函数*和*区域*。

- 以下示例展示如何使用 Proton 分析一个简单的 Python 函数。

```python
import triton.profiler as proton

# name: profile 数据的输出路径
# context: 标注每个内核上下文的方法，支持 "shadow" 和 "python" 两种模式
session_id = proton.profile(func, name="profile_name", context="python")(args)
```

- 以下示例展示如何在 Python 代码中分析一个区域。

```python
session_id = proton.start(name="profile_name", context="python")
...
# 跳过某个区域
proton.deactivate(session_id)
...
# 恢复分析
proton.activate(session_id)
...
# 输出 profile 数据并结束分析
proton.finalize()
```

### Scope

与 *python* 上下文提供文件、函数和行号不同，*shadow* 上下文提供代码中注解区域的标签。以下示例展示如何使用 *shadow* 上下文。

```python
import triton.profiler as proton

session_id = proton.start(name="profile_name", context="shadow")

with proton.scope("test0"):
    with proton.scope("test1"):
        foo[1,](x, y)
with proton.scope("test2"):
    foo[1,](x, y)

...
proton.finalize()
```

*scope* 工具也支持灵活的度量指标，通过字典提供指标名到数值（int 或 float）的映射。Proton 会聚合每个 scope 的指标并写入 profile 数据。

```python
with proton.scope("test0", {"bytes": 1000}):
    with proton.scope("test1", {"bytes": 2000}):
        foo[1,](x, y)
with proton.scope("test2", {"bytes": 3000}):
    foo[1,](x, y)
```

### Hook

```python
import triton.profiler as proton

# hook="triton" 启用 launch_metadata 函数，在内核启动前提供元数据
proton.start("profile_name", hook="triton")

def metadata_fn(grid, metadata, args):
    return {"name": "<kernel_name>", "flops8": 1.0}

@triton.jit(launch_metadata=metadata_fn)
def foo(x, y):
    tl.store(y, tl.load(x))
```

`metadata_fn` 在内核启动前调用，返回字典包含以下可选字段：

```python
name: str    # 内核名称
flops8: float   # 8 位浮点运算次数
flops16: float  # 16 位浮点运算次数
flops32: float  # 32 位浮点运算次数
flops64: float  # 64 位浮点运算次数
bytes: int      # 预期传输的字节数
```

### 命令行

Proton 可作为命令行工具使用。

```bash
proton [options] script.py [script_args] [script_options]
proton [options] pytest [pytest_args] [script_options]
python -m triton.profiler.proton [options] script.py [script_args] [script_options]
```

命令行模式下，`proton.start` 和 `proton.finalize` 会自动调用，脚本中的同名调用会被忽略。

注意：当前 Triton-Ascend 不提供 `proton` CLI 工具，请使用 Python API（`proton.start` / `proton.finalize`）。

### 可视化 Profile 数据

默认 profile 输出为 JSON 格式，可通过 Hatchet 读取。

```bash
pip install llnl-hatchet
proton-viewer -m time/s <profile.hatchet>
```

更多选项：

```bash
proton-viewer -h
```

### Ascend NPU 后端

Proton 在 Ascend NPU 上支持两种分析后端：**instrumentation** 和 **npu-native**。

#### Instrumentation 模式

Instrumentation 模式通过 Triton 内置插桩分析内核执行，提供 per-warp 粒度的周期计数。在 Ascend NPU 上自动检测为默认后端。

```python
import triton.profiler as proton
from triton.profiler import Default

# 自动检测：Ascend 上默认使用 instrumentation 模式
proton.start(name="profile_name", context="shadow")

# 显式指定 backend="npu"
proton.start(name="profile_name", context="shadow", backend="npu")

# 或使用通用名称 "instrumentation"
proton.start(name="profile_name", context="shadow", backend="instrumentation")

# 配置 buffer_size、sample_every_n
proton.start(name="profile_name", context="shadow", backend="npu",
             mode=Default(buffer_size=8192, sample_every_n=4))
```

通过 `InstrumentationMode` 支持以下选项：

- `buffer_size`（int）：每个内核的分析缓冲区大小（字节）。0 表示默认值 4096。scope 较多时应增大以避免溢出。
- `optimizations`（List[Optimize]）：优化标记。当前 Ascend 仅实现 `time_shift`（补偿 `GetSysCntOp` + `PipeBarrier` 开销，AI Core 约 12 周期）。其他标记 API 已定义但尚未实现。
- `sample_every_n`（int）：每 N 个 block 采样一次。1 = 全部采样（默认），2 = 每隔一个采样，以此类推。减少大 grid 的分析内存和开销。

#### NPU-Native 模式

npu-native 模式使用 Ascend ACL 分析 API（`aclprofInit/Start/Stop`）从 CANN profiling 数据中采集内核执行时间和 AI Core/Vector 硬件指标。

```python
import triton.profiler as proton

proton.start(name="profile_name", context="shadow", backend="npu-native")
```

npu-native 模式采集以下硬件指标：

| 指标 | 说明 |
|--------|-------------|
| `aic_mac_ratio` | AI Core MAC 单元利用率 |
| `aic_scalar_ratio` | AI Core Scalar 单元利用率 |
| `aic_mte1_ratio` | AI Core MTE1（内存传输引擎 1）比率 |
| `aic_mte2_ratio` | AI Core MTE2（内存传输引擎 2）比率 |
| `aic_fixpipe_ratio` | AI Core FixPipe 利用率 |
| `aic_icache_miss_rate` | AI Core 指令缓存缺失率 |
| `aiv_vec_ratio` | AI Vector 计算单元利用率 |
| `aiv_scalar_ratio` | AI Vector Scalar 单元利用率 |
| `aiv_mte1_ratio` | AI Vector MTE1 比率 |
| `aiv_mte2_ratio` | AI Vector MTE2 比率 |
| `aiv_mte3_ratio` | AI Vector MTE3 比率 |
| `aiv_icache_miss_rate` | AI Vector 指令缓存缺失率 |

可通过 `PROTON_ASCEND_OUTPUT_PATH` 环境变量配置分析数据输出路径（默认 `/tmp/ascend_profiling`）：

```bash
PROTON_ASCEND_OUTPUT_PATH=/data/profiling_output python your_script.py
```

## 已知问题

- Ascend NPU 分析输出路径

npu-native 模式将分析数据存储在 `PROTON_ASCEND_OUTPUT_PATH` 指定的目录（默认 `/tmp/ascend_profiling`），请确保目录存在且有足够磁盘空间。

- Ascend NPU msprof 依赖

npu-native 模式需要安装 CANN 工具包，并设置 `ASCEND_HOME_PATH` 或 `ASCEND_TOOLKIT_HOME` 环境变量以导入 msprof 数据。否则内核名称无法解析，硬件指标不可用。

- Ascend NPU 设备可见性

建议使用 `ASCEND_RT_VISIBLE_DEVICES` 环境变量控制 Ascend NPU 的设备可见性。使用 `CUDA_VISIBLE_DEVICES` 或其他非 Ascend 设备环境变量可能导致异常。

- Ascend NPU instrumentation 模式：同进程多配置不支持

由于 Ascend 驱动限制（`rtFunctionRegister` 不支持重复注册相同内核符号名的新二进制），在同一进程中运行多个不同配置的分析会话（如不同的 `buffer_size` 或 `sample_every_n`）可能产生错误结果。请在独立的进程或脚本调用中运行每个配置。
