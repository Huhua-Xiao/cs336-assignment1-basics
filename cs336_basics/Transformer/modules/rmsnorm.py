import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
import math
from einops import reduce


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model, **factory_kwargs))
        init.ones_(self.weight)

    #     self.reset_parameters()
    
    # def reset_parameters(self):
    #     init.ones_(self.weight)
    
    # def get_rms(self, input: Tensor):
    #     ms = reduce(input**2, "... d_model -> ... 1", "mean")
    #     return torch.sqrt(ms + self.eps)

    def forward(self, input: Tensor):
        input = input.to(torch.float32)
        ms = reduce(input**2, "... d_model -> ... 1", "mean")
        rms = torch.sqrt(ms + self.eps)
        normed = input / rms
        return normed * self.weight