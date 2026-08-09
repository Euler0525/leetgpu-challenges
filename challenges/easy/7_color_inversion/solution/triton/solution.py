import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(image, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    pixel_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pixel_offsets < width * height

    base = pixel_offsets * 4

    r = 255 - tl.load(image + base + 0, mask=mask)
    g = 255 - tl.load(image + base + 1, mask=mask)
    b = 255 - tl.load(image + base + 2, mask=mask)

    tl.store(image + base + 0, r, mask=mask)
    tl.store(image + base + 1, g, mask=mask)
    tl.store(image + base + 2, b, mask=mask)


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](image, width, height, BLOCK_SIZE)
