import torch
import triton
import triton.language as tl


@triton.jit
def matrix_copy_kernel(
    a_ptr, b_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N ** 2

    a_value = tl.load(a_ptr + offsets, mask=mask)
    tl.store(b_ptr + offsets, a_value, mask=mask)


# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    if N == 0:
        return

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N ** 2, BLOCK_SIZE), )
    matrix_copy_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)
