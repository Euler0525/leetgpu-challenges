import torch
import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def conv2d_kernel(
    input_ptr, kernel_ptr, output_ptr,
    input_rows: tl.constexpr,
    input_cols: tl.constexpr,
    kernel_rows: tl.constexpr,
    kernel_cols: tl.constexpr,
    output_rows: tl.constexpr,
    output_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_rows * output_cols

    output_row = offsets // output_cols
    output_col = offsets % output_cols

    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for kr in range(kernel_rows):
        for kc in range(kernel_cols):
            x = tl.load(input_ptr + (output_row + kr) * input_cols +
                        (output_col + kc), mask=mask, other=0.0)
            k = tl.load(kernel_ptr + kr * kernel_cols + kc)

            acc += x * k

    tl.store(output_ptr + offsets, acc, mask=mask)


def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    grid = (triton.cdiv(output_rows * output_cols, BLOCK_SIZE), )

    conv2d_kernel[grid](
        input, kernel, output,
        input_rows=input_rows,
        input_cols=input_cols,
        kernel_rows=kernel_rows,
        kernel_cols=kernel_cols,
        output_rows=output_rows,
        output_cols=output_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
