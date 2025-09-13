import torch
from torch import Tensor
from einops import einsum
from .attention.MultiheadSelfAttention import MultiheadSelfAttention
from .modules.swiglu import SwiGLU
from .modules.rmsnorm import RMSNorm

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device

        self.attention = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff, device=device)

        self.norm1 = RMSNorm(d_model, device=device)
        self.norm2 = RMSNorm(d_model, device=device)

    def forward(self, x: Tensor):
        batch, seq_len, _ = x.shape
        
        token_positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch, -1)

        self.x1 = x
        self.x = self.norm1(x)
        self.x = self.attention(self.x, token_positions) + self.x1

        self.x2 = self.x
        self.x = self.norm2(self.x)
        ffn_output = self.ffn(self.x) + self.x2
        return ffn_output
       