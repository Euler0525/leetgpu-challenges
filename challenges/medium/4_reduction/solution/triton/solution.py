import torch
import triton
import triton.language as tl


@triton.jit
def reduce_stage1_kernel(
    input_ptr, partial_ptr, N, BLOCK_SIZE: tl.constexpr, NUM_PARTIALS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for base in tl.range(0, N, NUM_PARTIALS * BLOCK_SIZE):
        idx = base + offsets
        x = tl.load(input_ptr + idx, mask=idx < N, other=0.0)
        acc += x.to(tl.float32)

    partial_sum = tl.sum(acc, axis=0)
    tl.store(partial_ptr + pid, partial_sum)


@triton.jit
def reduce_stage2_kernel(
    partial_ptr, output_ptr, num_partials, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(partial_ptr + offsets, mask=offsets < num_partials, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(output_ptr, result)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    if N == 0:
        output.zero_()
        return

    BLOCK_SIZE = 1024

    num_partials = min(256, triton.cdiv(N, BLOCK_SIZE))
    partial_sums = torch.empty(num_partials, device=input.device, dtype=torch.float32)

    reduce_stage1_kernel[(num_partials,)](
        input,
        partial_sums,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_PARTIALS=num_partials,
        num_warps=8,
    )

    stage2_block = triton.next_power_of_2(num_partials)

    reduce_stage2_kernel[(1,)](
        partial_sums,
        output,
        num_partials,
        BLOCK_SIZE=stage2_block,
        num_warps=1,
    )
