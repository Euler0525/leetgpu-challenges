import torch
import torch.nn.functional as F


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    x1 = input[:N//2]
    x2 = input[N//2:]
    torch.mul(x1 * F.sigmoid(x1), x2, out=output)
