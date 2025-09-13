import torch
from torch import Tensor
from typing_extensions import TypeAlias
from jaxtyping import Float, Int
from einops import einsum
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


def gradient_clipping(parameters, max_l2_norm:float):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
           total_norm += p.grad.data.norm() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_l2_norm:
        clip_value = max_l2_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_value)


def lr_cosine_schedule(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,):

    if it < warmup_iters:
       return max_learning_rate * it / warmup_iters

    if it > cosine_cycle_iters:
        return min_learning_rate
    
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    factor = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * factor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int, out):

    checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)


def load_checkpoint(src,  model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]