import torch
import torch.nn as nn
import torch.nn.functional as F

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


# next part is gating network
class Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_experts,
        k=2,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.W_g = nn.Linear(dim, num_experts, bias=False)
        self.k = k

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.W_g(x)
        values, index = torch.topk(logits, self.k, dim=-1)
        topk_probs = F.softmax(values)
        # there is more things we can do right now
        # expert capacity
        return topk_probs

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
