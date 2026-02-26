import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# first let's implement a EXPERT
class Expert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_exp: int,
    ):
        super().__init__()
        self.W1 = nn.Linear(input_dim, n_exp, bias=False)
        self.W2 = nn.Linear(n_exp, input_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.W2(F.silu(self.W1))


class Router(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, top_k: int):
        super().__init__()
        self.num_experts=num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    def forward(self, x):
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return top_indices, top_weights, gate_probs


class MoE(nn.Module):

    def __init__(self,
                 dim,
                 num_experts=8,
                 top_k=2
                 ):
        super().__init__()
        self.router = Router(
                dim=dim,
                num_experts=num_experts,
                k=top_k,
                )
        self.expert = Expert(
                input_dim=dim,
                n_exp=num_experts
                )
        self.n_experts = num_experts
    def forward(self, x: torch.Tensor):
        weights = self.router(x)
        y = torch.zeros_like(x)
        for i in range(self.n_experts):
            mask = weights[:, e] > 0
            if not mask.any():
                continue
            x_e = x[mask]
            y_e = self.expert(e, x_e)
            y[mask] += y_e * weights[mask, e].unsqeeze(-1)
        return y


