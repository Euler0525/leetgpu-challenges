import torch


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    image = image.view(-1, 4)
    image[:, 0:3] = 255 - image[:, 0:3]
