import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Expert(nn.Module):
    def __init__(
        self,
        dim: int,
        d_ff: int
    ):
        super().__init__()
        self.W1 = nn.Linear(dim, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, dim, bias=False)
        self.W3 = nn.Linear(dim, d_ff, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.W2(F.silu(self.W1(x)) * self.W3(x))

# Noisy TOP-K
class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        self.noise_linear = nn.Linear(d_model, num_experts)

    def forward(self, x):
        logits = self.gate(x)
        if self.training:
            noise = self.noise_linear(x)
            noise_std = F.softplus(noise)
            noisy_logits = logits + (torch.randn_like(logits) * noise_std)
        else:
            noisy_logits = logits
        top_k_logits, indices = torch.topk(noisy_logits, self.top_k, dim=1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class MoE(nn.Module):
    def __init__(self,
                 dim,
                 hidden_size,
                 num_experts=8,
                 top_k=2
                 ):
        super().__init__()
        self.router = Router(
                d_model=dim,
                num_experts=num_experts,
                top_k=top_k
                )
        self.expert = nn.ModuleList([Expert(
                dim=dim,
                d_ff=hidden_size
                ) for _ in range(num_experts)])
        self.n_experts = num_experts

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.router(x)
        y = torch.zeros_like(x)
        for i in range(self.n_experts):
            expert = self.expert[i]
            idx, top = torch.where([indices == i])
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        return y.view(shape)


