import torch
import torch.nn.functional as F

# X, Y are tensors on the GPU


def solve(X: torch.Tensor, Y: torch.Tensor, N: int):
    Y.copy_(F.sigmoid(X))
