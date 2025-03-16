import torch
from torch import nn
from torch.nn import functional as F

def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) module.

    Args:
        dim (int): The input dimension.
        affine (bool, optional): If True, apply an affine transformation to the normalized output.
            Default is True.

    Attributes:
        scale (float): The scaling factor for the normalized output.
        gamma (torch.Tensor or float): The learnable parameter for the affine transformation.

    """

    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.0

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale