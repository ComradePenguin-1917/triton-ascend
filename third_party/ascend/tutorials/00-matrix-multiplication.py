"""
Matrix Multiplication
===============
"""
import os
import pathlib

import triton
import triton.language as tl
import triton.profiler.language as pl
import triton.profiler as proton
import torch
import torch_npu

DEV = "npu"
activation = "leaky_relu_custom"

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = triton.runtime.driver.active.get_current_device()


def get_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
):
    with pl.scope("matmul-kernel"):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        GROUP_SIZE_M: tl.constexpr = 1
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        with pl.scope("matmul-initialization"):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            # ----------------------------------------------------------
            # Create pointers for the first blocks of A and B.
            # We will advance this pointer as we move in the K direction
            # and accumulate
            # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
            # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
            # See above `Pointer Arithmetic` section for details
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs_base = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs_base = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            msk_m = offs_am < M
            msk_n = offs_bn < N

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        with pl.scope("matmul-compute"):
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a_ptrs = a_ptrs_base + k * BLOCK_SIZE_K * stride_ak
                b_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * stride_bk

                with pl.scope("matmul-load-a"):
                    a = tl.load(
                        a_ptrs,
                        mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                        other=0.0,
                    )

                with pl.scope("matmul-load-b"):
                    b = tl.load(
                        b_ptrs,
                        mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
                        other=0.0,
                    )

                # We accumulate along the K dimension.
                with pl.scope("matmul-dot"):
                    accumulator = tl.dot(a, b, accumulator)
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        # Original vector operations
        # if ACTIVATION == "leaky_relu_custom":
        #     accumulator = leaky_relu_custom(accumulator)
        # c = accumulator.to(tl.float16)

        # # -----------------------------------------------------------
        # # Write back the block of the output matrix C with masks.
        # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        # tl.store(c_ptrs, c, mask=c_mask)
        # Comment out the following lines to enable split the workload to two vector cores
        with pl.scope("matmul-output"):
            SUB_BLK_M: tl.constexpr = BLOCK_SIZE_M // 2
            for s in tl.parallel(0, 2, bind_sub_block=True):
                with pl.scope("matmul-extract-slice"):
                    vec_sub_blk = tl.extract_slice(
                        accumulator, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1)
                    )

                with pl.scope("matmul-leaky-relu"):
                    if ACTIVATION == "leaky_relu_custom":
                        vec_sub_blk = leaky_relu_custom(vec_sub_blk)

                c_sub_blk = vec_sub_blk.to(tl.float16)
                # -----------------------------------------------------------
                # Write back the block of the output matrix C with masks.
                offs_cm = pid_m * BLOCK_SIZE_M + s * SUB_BLK_M + tl.arange(0, SUB_BLK_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                tl.store(c_ptrs, c_sub_blk, mask=c_mask)


# We can fuse `leaky_relu_custom` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu_custom(x):
    return tl.where(x >= 0, x, 0.01 * x) + 1.0


def torch_matmul(a, b, activation=""):
    c = torch.matmul(a, b)
    if activation == "leaky_relu_custom":
        c = torch.where(c >= 0, c, 0.01 * c) + 1.0
    return c


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    tmp_path = pathlib.Path(os.getcwd())
    temp_file = tmp_path / "matmul.json"
    proton.start(str(temp_file.with_suffix("")), backend="npu")
    matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        ACTIVATION=activation,  #
    )
    proton.finalize()

    return c


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).
torch.npu.set_device(1)
torch.manual_seed(0)
a = torch.randn((512, 512), device=DEV, dtype=torch.float16)
b = torch.randn((512, 512), device=DEV, dtype=torch.float16)
triton_output = matmul(a, b, activation)
torch_output = torch_matmul(a, b, activation)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
