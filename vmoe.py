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


# we are going to follow Group-level top-2 gating with auxiliary loss
# https://arxiv.org/pdf/2006.16668
class Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_experts,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.,
        k=2,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.W_g = nn.Linear(dim, num_experts, bias=False)
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.k = k

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.W_g(x)
        S = x.shape[0] # size of x
        E = self.num_experts
        gates = F.softmax(logits, dim=-1)
        mean_gates_per_expert = gates.mean(dim=0)
        combine_weights = torch.zeros_like(gates)
        expert_counts = torch.zeros(E, device=x.device)
        if self.training:
            capacity_factor = self.capacity_factor_train
        else:
            capacity_factor = self.capacity_factor_eval
        expert_capacity = int(capacity_factor * S / E)

        for s in range(S):
            g1, e1, g2, e2 = torch.topk(gates, k=self.k, dim=-1)
            g1 = g1/(g1+g2)
            e = e1[s]
            if expert_counts[e] < expert_capacity:
                combine_weights[s, e] = g1[s]
                expert_counts[e] += 1
        for s in range(S):
            g1, e1, g2, e2 = torch.topk(gates, k=self.k, dim=-1)
            g2 = g1/(g1+g2)
            rnd = torch.rand(1, device=x.device).item()
            e = e2[s]
            if expert_counts[e] < expert_capacity and rnd < 2 * g2:
                combine_weights[s, e] = g2[s]
                expert_counts[e] += 1
        return combine_weights


    def load_balancing_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.num_experts)
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0,1))
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0,1))
        loss = torch.sum(prob_per_expert, tokens_per_expert)
        return loss


class MoE(nn.Module):

    def __init__(self,
                 dim,
                 num_experts=8,
                 top_k=2
                 ):
        super().__init__()
        self.router = Gating(
                dim=dim,
                num_experts=num_experts,
                k=top_k,
                )
        self.expert = Expert(
                input_dim=dim,
                n_exp=num_experts
                )
    def forward(self, x):
        B, C, d = x.size()
        num_tokens = (B * C)
