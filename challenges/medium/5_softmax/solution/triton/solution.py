import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel(input_ptr, max_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    input_value = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
    tl.atomic_max(max_ptr, tl.max(input_value))


@triton.jit
def sumexp_kernel(input_ptr, max_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    max_value = tl.load(max_ptr)

    input_value = tl.load(input_ptr + offsets, mask=mask)
    z = tl.where(mask, tl.exp(input_value - max_value), 0.0)
    tl.atomic_add(output_ptr, tl.sum(z))


@triton.jit
def softmax_kernel(input_ptr, max_ptr, sumexp_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    max_value = tl.load(max_ptr)
    sumexp_value = tl.load(sumexp_ptr)

    input_value = tl.load(input_ptr + offsets, mask=mask)

    y = tl.exp(input_value - max_value) / sumexp_value
    tl.store(output_ptr + offsets, y, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    if N == 0:
        return

    assert input.is_contiguous()
    assert output.is_contiguous()
    assert input.device == output.device
    assert N <= input.numel()
    assert N <= output.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    max_buf = torch.full((1,), -float("inf"), device=input.device, dtype=torch.float32, )
    sumexp_buf = torch.zeros((1,), device=input.device, dtype=torch.float32, )

    max_kernel[grid](input, max_buf, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, )
    sumexp_kernel[grid](input, max_buf, sumexp_buf, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, )
    softmax_kernel[grid](input, max_buf, sumexp_buf, output, N,
                         BLOCK_SIZE=BLOCK_SIZE, num_warps=4, )
