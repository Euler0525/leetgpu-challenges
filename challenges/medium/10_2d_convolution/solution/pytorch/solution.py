import torch
import torch.nn.functional as F

# input, kernel, output are tensors on the GPU


def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    x = input.view(1, 1, input_rows, input_cols)  # [N, C, H, W]
    # [out_channels, in_channels, kernel_height, kernel_width]
    k = kernel.view(1, 1, kernel_rows, kernel_cols)
    y = F.conv2d(x, k, stride=1, padding=0)

    # output_rows = input_rows - kernel_rows + 1
    # output_cols = input_cols - kernel_cols + 1
    output.copy_(y.reshape(-1))
