import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from typing import Tuple
from typing_extensions import TypeAlias
from jaxtyping import Float, Int
from einops import reduce, einsum, rearrange
import math
from ..modules.linear import Linear
from ..functions import scaled_dot_product_attention



class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = None):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.q_proj_linear = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj_linear = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj_linear = Linear(self.d_model, self.num_heads * self.d_k)
        self.out_proj_linear = Linear(self.num_heads * self.d_k, self.d_model)


    def forward(self, x: Tensor):
        batch, seq_len, d_model = x.shape
        Q = self.q_proj_linear(x)
        K = self.k_proj_linear(x)
        V = self.v_proj_linear(x)

        Q = rearrange(Q, "batch seq_len (num_heads d_k) -> batch num_heads seq_len d_k", num_heads = self.num_heads)
        K = rearrange(K, "batch seq_len (num_heads d_k) -> batch num_heads seq_len d_k", num_heads = self.num_heads)
        V = rearrange(V, "batch seq_len (num_heads d_k) -> batch num_heads seq_len d_k", num_heads = self.num_heads)

        dev = x.device
        self.mask = torch.tril(torch.ones(seq_len, seq_len)).to(dev)

        attention = scaled_dot_product_attention(Q, K, V, self.mask)
        output = rearrange(attention, "batch num_heads seq_len d_k -> batch seq_len (num_heads d_k)")
        return self.out_proj_linear(output)



