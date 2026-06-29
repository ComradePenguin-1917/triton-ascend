# Proton 教程脚本说明

## 前置条件

- 已安装 Triton-Ascend（含 Proton 支持）
- 已激活 Triton conda 环境
- Ascend NPU 设备可用

## 脚本列表

### `01_instrumentation.py` — Instrumentation 模式示例

Proton 最常用的 profiling 模式。通过 `pl.scope` 在内核代码中标记 profiling 区域，获取每个 warp 的 cycle 计数。

```bash
# 示例 1：默认配置（所有 block 采样，4096 字节 buffer）
python 01_instrumentation.py -e 1

# 示例 2：采样模式（每 4 个 block 采样一次，减少 profiling 开销）
python 01_instrumentation.py -e 2

# 示例 3：大 buffer（8192 字节，避免 buffer 溢出）
python 01_instrumentation.py -e 3

# 示例 4：Python 级别 proton.scope 外层分组
python 01_instrumentation.py -e 4
```

输出：`instrumentation_*.chrome_trace` 文件，可用 Chrome `chrome://tracing` 打开。

### `02_npu-native.py` — NPU-Native 模式示例

使用 CANN ACL profiling API 采集硬件级指标（MAC 利用率、cache miss 率等），不需要修改内核代码。

```bash
python 02_npu-native.py
```

可选环境变量：
- `PROTON_ASCEND_OUTPUT_PATH`：profiling 输出目录（默认 `/tmp/ascend_profiling`）

注意：此模式不支持 `pl.scope` 的 per-scope 数据，使用 `proton.scope`（Python 级别）进行区域分组。

### `03_overhead.py` — Instrumentation 开销测试

对比无 profiling（plain）和有 profiling（instrumented）两种内核的执行时间，用于评估 proton instrumentation 的性能开销。

```bash
# 示例 1：默认 buffer 配置
python 03_overhead.py -e 1

# 示例 2：采样模式
python 03_overhead.py -e 2
```

### `run_msprof_overhead.sh` — msprof 集成开销测试（Shell 脚本）

通过 CANN 的 `msprof op` 工具运行 `03_overhead.py`，直接提取 `Task Duration` 来测量开销，无需手工解析 CSV。

```bash
bash run_msprof_overhead.sh 1   # 示例 1
bash run_msprof_overhead.sh 2   # 示例 2
```

注意：
- 需要 CANN 环境（`msprof` 命令可用）
- 当前目录要求权限为 `755`（不允许 group/other writable），否则 `msprof` 会报错
- 脚本会自动清理临时文件
