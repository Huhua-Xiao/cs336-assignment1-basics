import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
import math
from einops import reduce, einsum, rearrange



class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k:int, max_seq_len:int, device: torch.device):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device) / self.d_k))
        position = torch.arange(0, max_seq_len, device=device)
        angles = position[:, None] * self.inv_freq[None, :]
        cos = angles.cos()
        sin = angles.sin()
        R = torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)
        self.register_buffer("R", R, persistent=False)


    def forward(self, x: Tensor, token_pos: Tensor):
    #    print("x shape:", x.shape)
    #    print("token_pos shape:", token_pos.shape)
    #    print("R shape:", self.R.shape)
    # x_split = rearrange(x, "... (d_kk r2) -> ... d_kk r2", r2 = 2)
        x_split = rearrange(x, "... (d_pair r2) -> ... d_pair r2", r2 = 2)
        rot = self.R[token_pos]
        print("rot shape:", rot.shape)
        output = einsum(rot, x_split, "... s d_pair r1 r2, ... h s d_pair r2 -> ... h s d_pair r1")
        result = rearrange(output, "... h s d_pair r1 -> ... h s (d_pair r1)")
        return result

# x shape: torch.Size([4, 12, 64])
# token_pos shape: torch.Size([12])
# R shape: torch.Size([12, 32, 2, 2])
# rot shape: torch.Size([12, 32, 2, 2])
# FAILED

# token_pos = (batch, seq_len) = (1, 12)
# R = (max_seq_len, d_k/2, 2, 2)
# rot = (batch, seq_len, d_k/2, 2, 2) = (1, 12, 32, 2, 2)
# x = (batch, num_heads, seq_len, d_k) = (1, 4, 12, 64)
# x_split = (batch, num_heads, seq_len, d_k/2, 2) = (1, 4, 12, 32, 2)

