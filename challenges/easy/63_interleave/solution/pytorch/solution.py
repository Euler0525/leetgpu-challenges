import torch


# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    # output[::2] = A, output[1::2] = B
    output[:] = torch.stack((A, B), dim=1).reshape(-1)
