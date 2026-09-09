import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    m = tl.full((1, ), -float("inf"), tl.float32)  # max_value
    l = tl.zeros((1, ), tl.float32)  # sumexp_value
    for start in tl.range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        input_value = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)

        block_m = tl.max(input_value, axis=-1)
        m_new = tl.maximum(block_m, m)
        alpha = tl.exp(m - m_new)

        block_l = tl.sum(tl.exp(input_value - m_new), axis=0)
        l = l * alpha + block_l
        m = m_new

    for start in tl.range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        input_value = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)

        y = tl.exp(input_value - m) / l
        tl.store(output_ptr + offsets, y, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    if N == 0:
        return

    assert input.is_contiguous()
    assert output.is_contiguous()
    assert input.device == output.device
    assert N <= input.numel()
    assert N <= output.numel()

    BLOCK_SIZE = 1024
    grid = (1, )

    online_softmax_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, )
