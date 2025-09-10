import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from typing import Tuple
from typing_extensions import TypeAlias
from jaxtyping import Float, Int
from einops import reduce, einsum
import math


def softmax(in_features: Tensor, dim: int):
    exp_in = torch.exp(in_features - torch.max(in_features, dim=dim, keepdim=True).values)
    return exp_in / torch.sum(exp_in, dim=dim, keepdim=True)    


def cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    logits = inputs
    B, V = logits.shape
    logsumexp = torch.logsumexp(logits, dim=1)  # (B,)
    logits_y = logits[torch.arange(B), targets]  # (B,)
    loss = (logsumexp - logits_y).mean()  # ()
    return loss


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
):
    d_k = Q.shape[-1]
    QK = einsum(Q, K, " ... queries d_k, ... values d_k -> ... queries values")
    scores = QK / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    
    attention_weights = softmax(scores, dim=-1)
    output = einsum(attention_weights, V, " ... queries values, ... values d_v -> ... queries d_v")
    return output