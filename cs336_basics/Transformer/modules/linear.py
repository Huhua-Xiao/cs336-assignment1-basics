import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from einops import einsum
import math


class Linear(torch.nn.Module):
    def __init__(self, in_feature: int, out_feature: int, device: torch.device=None, dtype: torch.dtype=None) -> None:
        super().__init__()

        factory_kwargs ={"device": device, "dtype": dtype}

        self.in_feature = in_feature
        self.out_feature = out_feature

        self.weight = Parameter(torch.ones(out_feature, in_feature, **factory_kwargs))
        std = math.sqrt(2.0 / (self.in_feature + self.out_feature))
        init.trunc_normal_(self.weight, mean = 0, std = std, a = -3*std, b= 3*std)

    #     self.reset_parameters()
    
    # def reset_parameters(self):
    #     std = math.sqrt(2.0 / (self.in_feature + self.out_feature))
    #     init.trunc_normal_(self.weight, mean = 0, std = std, a = -3*std, b= 3*std)

    def forward(self, input: Tensor):
        return einsum(input, self.weight, "... d_in, d_out d_in -> ... d_out")
