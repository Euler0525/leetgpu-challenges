import torch
import triton
import triton.language as tl


@triton.jit
def rgb_to_grayscale_kernel(input, output, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < width * height

    r = tl.load(input + offsets * 3 + 0, mask=mask)
    g = tl.load(input + offsets * 3 + 1, mask=mask)
    b = tl.load(input + offsets * 3 + 2, mask=mask)

    output_value = 0.299 * r + 0.587 * g + 0.114 * b
    tl.store(output + offsets, output_value, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, width: int, height: int):
    total_pixels = width * height
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_pixels, BLOCK_SIZE),)
    rgb_to_grayscale_kernel[grid](input, output, width, height, BLOCK_SIZE)
