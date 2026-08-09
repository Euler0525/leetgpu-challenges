import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_am,
    stride_an,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_ck,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets_n = tl.arange(0, BLOCK_N)

    a_ptrs = a + offsets_m[:, None] * stride_am + offsets_n[None, :] * stride_an
    b_ptrs = b + offsets_n[:, None] * stride_bn + offsets_k[None, :] * stride_bk
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        a_values = tl.load(
            a_ptrs,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] + n_start < N),
            other=0.0,
        )
        b_values = tl.load(
            b_ptrs,
            mask=(offsets_n[:, None] + n_start < N) & (offsets_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(a_values, b_values, input_precision="ieee")
        a_ptrs += BLOCK_N * stride_an
        b_ptrs += BLOCK_N * stride_bn

    c_ptrs = c + offsets_m[:, None] * stride_cm + offsets_k[None, :] * stride_ck
    tl.store(c_ptrs, acc, mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K))


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K),)
    matrix_multiplication_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_an,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_ck,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
