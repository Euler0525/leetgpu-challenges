import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(input, kernel, output, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
    output_size = input_size - kernel_size + 1
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in tl.range(kernel_size):
        x = tl.load(input + offsets + i, mask=mask, other=0.0)
        k = tl.load(kernel + i)
        acc += x * k

    tl.store(output + offsets, acc, mask=mask)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE)
