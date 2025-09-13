import torch
from torch import Tensor
from einops import einsum
from .modules.linear import Linear
from .modules.rmsnorm import RMSNorm
from .transformer_block import TransformerBlock
from .modules.embedding import Embedding


class Transformer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int, vocab_size: int, num_layers: int, 
                device: torch.device=None,  theta: float = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.theta = theta
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.device = device

        self.transformer_layer = torch.nn.ModuleList([TransformerBlock(self.d_model, self.num_heads, self.d_ff, 
                                                                       self.context_length, self.theta, device=self.device) for _ in range(self.num_layers)])
        self.embedding = Embedding(self.vocab_size, self.d_model, device=self.device)
        self.norm = RMSNorm(self.d_model, device=self.device)
        self.output_linear = Linear(self.d_model, self.vocab_size, device=self.device)

    def forward(self, token_ids: Tensor):
        x = self.embedding(token_ids)
        for layer in self.transformer_layer:
           x = layer(x)
        x = self.norm(x)
        x = self.output_linear(x)
        return x
        



