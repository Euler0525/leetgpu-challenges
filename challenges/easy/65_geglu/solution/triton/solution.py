import torch
import triton
import triton.language as tl


@triton.jit
def geglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N // 2

    x1 = tl.load(input + offsets, mask=mask)
    x2 = tl.load(input + N // 2 + offsets, mask=mask)

    tl.store(output + offsets, x1 * 0.5 * x2 * (1.0 + tl.erf(x2 / tl.sqrt(2.0))), mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    geglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
