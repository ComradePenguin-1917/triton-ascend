# Proton - A Profiler for Triton

## Introduction

Proton is a lightweight profiler for Triton, designed to be used for code written in Python and to invoke underlying GPU/NPU kernels. Proton provides insightful information about the program context, metadata, and hardware performance metrics of the GPU/NPU kernels invoked.

## Installation

Proton is built as part of Triton-Ascend. Follow the [installation guide](../../docs/en/installation_guide.md) to build from source with Proton enabled (default).

## Usage

### Basic usage

More examples can be found in the [tutorials](tutorials) directory.

Proton can be used to profile *functions* and *regions* in Python code.

- The following examples demonstrate how to use Proton to profile a simple Python function.

```python
import triton.profiler as proton

# name: The path to the profile data
# context: The method used to annotate the context of each GPU kernel. Currently, "shadow" and "python" are supported.
session_id = proton.profile(func, name="profile_name", context="python")(args)
```

- The following examples demonstrate how to use Proton to profile a region in Python code.

```python
session_id = proton.start(name="profile_name", context="python")
...
# Skip a region
proton.deactivate(session_id)
...
# Restart profiling
proton.activate(session_id)
...
# Write out the profile data and finalize the profiler
proton.finalize()
```

### Scope

Unlike the *python* context that provide users with files, functions, and lines where the GPU kernels are invoked, the *shadow* context provides users with the annotated regions in the code. The following example demonstrates how to use the *shadow* context.

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

The *scope* utility also accepts flexible metrics, provided with a dictionary that maps from a string (metric name) to a value (int or float).
Proton will aggregate the metrics for each scope and write them to the profile data.
It is useful for users to understand the performance of the model at a high level.

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
from typing import NamedTuple

# hook: When hook="triton", it enables proton to invoke launch_metadata function before launching the GPU kernel
proton.start("profile_name", hook="triton")

def metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict
):
    return {"name": "<kernel_name>", "flops8": 1.0}

@triton.jit(launch_metadata=metadata_fn)
def foo(x, y):
    tl.store(y, tl.load(x))
```

The `metadata_fn` function is called before launching the GPU kernel to provide metadata for the GPU kernel, which returns a dictionary that maps from a string (metadata name) to a value (int or float).

Currently, **only the triton hook is supported**. In the dictionary returned by the `metadata_fn` function, we can supply the following keys:

```python
name: str  # The name of the kernel
flops8: float  # The number of 8-bit floating-point operations
flops16: float  # The number of 16-bit floating-point operations
flops32: float  # The number of 32-bit floating-point operations
flops64: float  # The number of 64-bit floating-point operations
bytes: int  # The number of bytes expected to be transferred
```

### Command line

Proton can be used as a command-line tool to profile Python scripts and Pytest tests.

```bash
proton [options] script.py [script_args] [script_options]
proton [options] pytest [pytest_args] [script_options]
python -m triton.profiler.proton [options] script.py [script_args] [script_options]
```

When profiling in the command line mode, the `proton.start` and `proton.finalize` functions are automatically called before and after the script execution. Any `proton.start` and `proton.finalize` functions in the script are ignored. Also, in the command line mode, only a single *session* is supported. Therefore, `proton.deactivate(session_id=1)` is invalid, while `proton.deactivate(session_id=0)` is valid.

Note: The `proton` CLI tool is not available in the current Triton-Ascend build. Use the Python API (`proton.start` / `proton.finalize`) instead.

### Visualizing the profile data

By default, proton profiles are in the *json* format and can be read by *Hatchet*. The following command visualizes the profile data on terminal.

```bash
pip install llnl-hatchet
proton-viewer -m time/s <profile.hatchet>
```

NOTE: `pip install hatchet` does not work because the API is slightly different.

More options can be found by running the following command.

```bash
proton-viewer -h
```

### Ascend NPU Backends

Proton supports two profiling backends on Ascend NPU: **instrumentation** and **npu-native**.

#### Instrumentation Mode

The instrumentation mode uses Triton's built-in instrumentation to profile kernel execution with per-warp granularity. It is the default backend when running on Ascend NPU (auto-detected).

```python
import triton.profiler as proton
from triton.profiler import Default

# Auto-detected: uses instrumentation mode on Ascend
proton.start(name="profile_name", context="shadow")

# Explicit: specify backend="npu" (same as auto-detection)
proton.start(name="profile_name", context="shadow", backend="npu")

# Or use the generic backend name "instrumentation"
proton.start(name="profile_name", context="shadow", backend="instrumentation")

# With mode options: buffer_size, sample_every_n
proton.start(name="profile_name", context="shadow", backend="npu",
             mode=Default(buffer_size=8192, sample_every_n=4))
```

The instrumentation mode supports the following options via `InstrumentationMode`:

- `buffer_size` (int): Per-kernel profiling buffer size in bytes. 0 means default (4096). Increase for kernels with many scopes to avoid overflow.
- `optimizations` (List[Optimize]): Optimization flags. Currently, only `time_shift` is implemented on Ascend (compensates for `GetSysCntOp` + `PipeBarrier` overhead, ~12 cycles on AI Core). The other flags are defined in the API but not yet implemented.
- `sample_every_n` (int): Profile every N-th block. 1 = all blocks (default), 2 = every 2nd block, etc. Reduces profiling memory and overhead for large grids.

#### NPU-Native Mode

The npu-native mode uses the Ascend ACL profiling API (`aclprofInit/Start/Stop`) to collect kernel execution time and AI Core/Vector hardware metrics from CANN profiling data.

```python
import triton.profiler as proton

proton.start(name="profile_name", context="shadow", backend="npu-native")
```

The npu-native mode collects the following hardware metrics for each kernel (when available):

| Metric | Description |
|--------|-------------|
| `aic_mac_ratio` | AI Core MAC unit utilization ratio |
| `aic_scalar_ratio` | AI Core Scalar unit utilization ratio |
| `aic_mte1_ratio` | AI Core MTE1 (memory transfer engine 1) ratio |
| `aic_mte2_ratio` | AI Core MTE2 (memory transfer engine 2) ratio |
| `aic_fixpipe_ratio` | AI Core FixPipe utilization ratio |
| `aic_icache_miss_rate` | AI Core instruction cache miss rate |
| `aiv_vec_ratio` | AI Vector compute unit utilization ratio |
| `aiv_scalar_ratio` | AI Vector scalar unit utilization ratio |
| `aiv_mte1_ratio` | AI Vector MTE1 ratio |
| `aiv_mte2_ratio` | AI Vector MTE2 ratio |
| `aiv_mte3_ratio` | AI Vector MTE3 ratio |
| `aiv_icache_miss_rate` | AI Vector instruction cache miss rate |

You can configure the output path for profiling data via the `PROTON_ASCEND_OUTPUT_PATH` environment variable (default: `/tmp/ascend_profiling`):

```bash
PROTON_ASCEND_OUTPUT_PATH=/data/profiling_output python your_script.py
```

## Known issues

- Ascend NPU profiling output path

The npu-native mode stores profiling data in the directory specified by `PROTON_ASCEND_OUTPUT_PATH` (default: `/tmp/ascend_profiling`). Ensure the directory exists and has sufficient disk space before profiling.

- Ascend NPU msprof dependency

The npu-native mode requires CANN toolkit to be installed and `ASCEND_HOME_PATH` or `ASCEND_TOOLKIT_HOME` environment variable to be set for msprof data import. Without this, kernel names may not be resolved and hardware metrics may be unavailable.

- Visible devices on Ascend NPUs

Environment variables `ASCEND_RT_VISIBLE_DEVICES` is recommended to control device visibility on Ascend NPUs. Using `CUDA_VISIBLE_DEVICES` or other non-Ascend device environment variables may cause unexpected behavior.

- Ascend NPU instrumentation mode: same-process multi-configuration not supported

Due to an Ascend driver limitation (`rtFunctionRegister` does not support re-registering a new binary with the same kernel symbol name), running multiple profiling sessions with different configurations (e.g., different `buffer_size` or `sample_every_n`) in the same process may cause incorrect results. Please run each configuration in a separate process or script invocation.
