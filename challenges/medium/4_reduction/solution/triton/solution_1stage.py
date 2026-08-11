import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["N"],
    reset_to_zero=["output"],
)
@triton.jit
def reduce(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    input_value = tl.load(input + offsets, mask=mask, other=0.0)
    tl.atomic_add(output, tl.sum(input_value))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.zero_()

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]), )
    reduce[grid](input, output, N)
