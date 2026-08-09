import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
):
    m = tl.program_id(axis=0)
    k = tl.program_id(axis=1)

    BLOCK_N: tl.constexpr = 256
    offsets = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_start in tl.range(0, N, BLOCK_N):
        n = n_start + offsets
        mask = n < N
        a_value = tl.load(a + m * stride_am + n * stride_an, mask=mask, other=0.0).to(tl.float32)
        b_value = tl.load(b + n * stride_bn + k * stride_bk, mask=mask, other=0.0).to(tl.float32)
        acc += a_value * b_value
    result = tl.sum(acc, axis=0)

    tl.store(c + m * stride_cm + k * stride_ck, result)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = (M, K)
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
    )
