import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc):
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)

    x = tl.load(input + stride_ir * row + stride_ic * col)
    tl.store(output + stride_or * col + stride_oc * row, x)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    grid = (rows, cols)
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc
    )
