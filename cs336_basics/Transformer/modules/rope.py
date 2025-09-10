import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
import math



class RoPE(torch.nn.module):
    def __init__(self, theta: float, d_model:int, max_seq_len:int, device: torch.device=None):
        super().__init__()

        self.theta = theta
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device

        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_model, 2, device=device) / self.d_model))
        position = torch.arange(0, max_seq_len, device=device)
        angles = position[:, None] * self.inv_freq[None, :]
        cos = angles.cos()
        sin = angles.sin()
        R = torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)
        self.register_buffer("R", R, persistent=False)


    def forward(self, x: Tensor, token_pos: Tensor):
        # x_split = x.rearrange(" ")
        # rotate = self.R[token_pos]
        # out_pair = 

