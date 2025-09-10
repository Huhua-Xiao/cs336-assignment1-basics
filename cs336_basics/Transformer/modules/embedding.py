import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
import math

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size:int, d_model: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        
        factory_kwargs = ({"device": device, "dtype": dtype})

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.weight = Parameter(torch.ones(vocab_size, d_model, **factory_kwargs))
        init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    #     self.reset_parameters()
    
    # def reset_parameters(self):
    #     init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: int):
        return self.weight[token_ids]