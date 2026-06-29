"""
Proton NPU-Native Mode on Ascend NPU
=====================================

NPU-native mode uses the Ascend ACL profiling API (aclprofInit/Start/Stop)
to collect kernel execution time and AI Core/Vector hardware metrics from
CANN profiling data.

When to use:
- You need hardware-level metrics (MAC utilization, cache miss rates, etc.)
- You want kernel-level timing without modifying kernel code (no pl.scope needed)
- You want to analyze AI Core pipeline utilization

Backend: `backend="npu-native"`

Note: pl.scope annotations inside kernels do NOT produce per-scope data in
this mode. Use proton.scope (Python level) for region grouping instead.

Environment:
- PROTON_ASCEND_OUTPUT_PATH: profiling output directory (default /tmp/ascend_profiling)
- ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME: CANN toolkit path (required for msprof import)

Usage:
    python 02_npu_native_mode.py
"""

import pathlib
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.profiler as proton

DEV = "npu"


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[:, None] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


def run_gemm_native(proton_name, **start_kwargs):
    M, N, K = 4096, 4096, 4096
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    DTYPE = torch.float16

    a = torch.randn((M, K), device=DEV, dtype=DTYPE)
    b = torch.randn((K, N), device=DEV, dtype=DTYPE)
    c = torch.empty((M, N), device=DEV, dtype=DTYPE)
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    out_path = pathlib.Path(proton_name)
    proton.start(str(out_path), context="shadow", **start_kwargs)

    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    proton.finalize()

    ref = torch.matmul(a, b)
    torch.npu.synchronize()
    match = torch.allclose(c, ref, rtol=1e-2, atol=1e-2)
    print(f"\n{'='*60}")
    print(f"Profile: {proton_name}")
    print(f"Result: {'PASS' if match else 'FAIL'}")
    print(f"{'='*60}")


def main():
    # ── Example 1: Basic npu-native profiling ──
    # No pl.scope needed. ACL profiling captures kernel-level timing
    # and hardware metrics automatically.
    run_gemm_native("npu_native_basic", backend="npu-native")

    # ── Example 2: With proton.scope for region grouping ──
    # proton.scope at Python level groups kernels into named regions
    # in the profiling output. This is the only way to add hierarchy
    # in npu-native mode (pl.scope inside kernels is ignored).
    M, N, K = 8192, 7168, 3072
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 128
    DTYPE = torch.float16

    a = torch.randn((M, K), device=DEV, dtype=DTYPE)
    b = torch.randn((K, N), device=DEV, dtype=DTYPE)
    c = torch.empty((M, N), device=DEV, dtype=DTYPE)
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    out_path = pathlib.Path("npu_native_scoped")
    proton.start(str(out_path), context="shadow", backend="npu-native")

    with proton.scope("warmup"):
        gemm_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

    with proton.scope("measured"):
        gemm_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

    proton.finalize()
    print(f"\nScoped npu-native profile saved to: {out_path}.hatchet")

    # ── Example 3: Custom output path via environment variable ──
    # By default, ACL profiling data goes to /tmp/ascend_profiling.
    # Set PROTON_ASCEND_OUTPUT_PATH to change this.
    # Example: PROTON_ASCEND_OUTPUT_PATH=/data/prof python 02_npu_native_mode.py
    import os
    current_output_path = os.environ.get("PROTON_ASCEND_OUTPUT_PATH", "/tmp/ascend_profiling")
    print(f"\nACL profiling output path: {current_output_path}")
    print("Set PROTON_ASCEND_OUTPUT_PATH to change the output directory.")


if __name__ == "__main__":
    main()
