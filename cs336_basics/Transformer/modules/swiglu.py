import torch
from torch import Tensor
from einops import einsum
from .linear import Linear


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device)
        self.w2 = Linear(d_ff, d_model, device)
        self.w3 = Linear(d_model, d_ff, device)
    
    def forward(self, x: Tensor):
        u = self.w1(x)
        g = self.w3(x)
        a = u * torch.sigmoid(u)
        h = einsum(g, a, "b s d_ff, b s d_ff -> b s d_ff")
        return self.w2(h)
