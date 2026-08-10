import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, width: int, height: int):
    coe = torch.tensor([0.299, 0.587, 0.114], device=input.device, dtype=input.dtype)
    output.copy_(torch.matmul(input.view(-1, 3), coe))
