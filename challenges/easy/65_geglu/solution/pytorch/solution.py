import torch
import torch.nn.functional as F


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    x1, x2 = input[:N // 2], input[N // 2:]
    output.copy_(torch.mul(x1, F.gelu(x2)))
