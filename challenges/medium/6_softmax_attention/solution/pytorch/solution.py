import torch
import torch.nn.functional as F


def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    scores = torch.matmul(Q, K.T) / (d ** 0.5)
    attention = F.softmax(scores, dim=-1)
    output.copy_(torch.matmul(attention, V))
