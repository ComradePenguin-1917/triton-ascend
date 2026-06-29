"""
Proton Instrumentation Mode on Ascend NPU
==========================================

Instrumentation mode inserts profiling code directly into the Triton kernel
via `pl.scope`, yielding per-warp cycle counts for each annotated region.

Backend: `backend="npu"` (default on Ascend) or `backend="instrumentation"`

Usage:
    python 01_instrumentation_mode.py [--example 1|2]

Examples:
    1: Default buffer, all blocks profiled
    2: Sampling (sample every 8th block)
"""

import argparse
import pathlib
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler import Default

DEV = "npu"

# ── Instrumented kernel (with pl.scope) ──────────────────────────────────────
@triton.jit
def gemm_kernel_instrumented(
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

    with pl.scope("init"):
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        with pl.scope("load-a"):
            a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        with pl.scope("load-b"):
            b = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        with pl.scope("dot"):
            accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    with pl.scope("store"):
        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[:, None] < N)
        tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


# ── Plain kernel (no pl.scope) ───────────────────────────────────────────────
@triton.jit
def gemm_kernel_plain(
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


def run_gemm(proton_name, **start_kwargs):
    M, N, K = 8192, 7168, 3072
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 128
    DTYPE = torch.float16

    a = torch.randn((M, K), device=DEV, dtype=DTYPE)
    b = torch.randn((K, N), device=DEV, dtype=DTYPE)
    c = torch.empty((M, N), device=DEV, dtype=DTYPE)
    c_ref = torch.empty((M, N), device=DEV, dtype=DTYPE)
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    gemm_kernel_plain[grid](
        a, b, c_ref,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_ref.stride(0), c_ref.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    out_path = pathlib.Path(proton_name)
    proton.start(str(out_path), data="trace", context="shadow", **start_kwargs)

    gemm_kernel_instrumented[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    proton.finalize(output_format="chrome_trace")

    torch.npu.synchronize()
    match = torch.allclose(c, c_ref, rtol=1e-2, atol=1e-2)

    trace_file = str(out_path) + ".chrome_trace"
    import json
    from collections import Counter
    with open(trace_file) as f:
        data = json.load(f)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    names = sorted({e["name"] for e in events if isinstance(e, dict) and e.get("name")})
    name_counts = Counter(e.get("name") for e in events if isinstance(e, dict) and e.get("name"))

    print(f"\n{'='*60}")
    print(f"Profile: {proton_name}")
    print(f"Result: {'Proton and Triton Baseline match' if match else 'Proton and Triton Baseline mismatch'}")
    print(f"Trace: {trace_file}")
    print(f"Total events: {len(events)}")
    print(f"Scope names: {names}")
    print(f"Event counts:")
    for name in sorted(name_counts.keys()):
        print(f"  {name}: {name_counts[name]}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Proton instrumentation mode examples on Ascend NPU")
    parser.add_argument("--example", "-e", type=int, choices=[1, 2], default=1,
                        help="Which example to run (1-2, default: 1)")
    args = parser.parse_args()

    if args.example == 1:
        print("Running Example 1: Auto-detected backend (default on Ascend)")
        run_gemm("instrumentation_auto", backend="npu")

    elif args.example == 2:
        print("Running Example 2: With sampling (every 8th block)")
        run_gemm("instrumentation_sampled", backend="npu",
                 mode=Default(sample_every_n=8))

if __name__ == "__main__":
    main()
