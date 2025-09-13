import torch
from torch import Tensor
from einops import einsum
from .modules.linear import Linear
from .attention.MultiheadSelfAttention import MultiheadSelfAttention
from .modules.swiglu import SwiGLU
from .modules.rmsnorm import RMSNorm
from .transformer_block import TransformerBlock

class Tranformerlm(torch.nn.module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device = torch.device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
    
        self.transformer_block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)
        