import torch
import torch.nn.functional as F

# input, kernel, output are tensors on the GPU


def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    x = input.view(1, 1, input_size)
    k = kernel.view(1, 1, kernel_size)

    with torch.no_grad():
        y = F.conv1d(x, k)
        output.copy_(y.view(-1))
