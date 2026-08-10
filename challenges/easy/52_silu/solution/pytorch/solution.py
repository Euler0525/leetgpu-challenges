import torch
import torch.nn.functional as F


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # output.copy_(F.silu(input))
    torch.mul(input, torch.sigmoid(input), out=output)
