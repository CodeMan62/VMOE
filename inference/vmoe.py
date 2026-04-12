import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#--------------------------------------------------------------------------------

class ModelConfig:
    d_model: int = 128
    d_c: int = ...
    n_heads: int = 4
    d_comp: int = ...
    max_seq_len: int = ...
    d_hr: int = ...
    d_rope: int = ...
    inter_dim: int = ...


class RmsNorm(nn.Module):
    def __init__(self, config: ModelConfig, eps=1e-5):
        super().__init__()
        self.d_model = config.d_model
        self.g = nn.Parameter(torch.ones(self.d_model))
        self.eps = eps
    def forward(self, x:torch.Tensor):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.g

class MLA(nn.Module):
    def __init__(self, config: ModelConfig):
        self.dim = config.d_model
        self.num_heads = config.n_heads
        self.d_heads = self.dim // self.num_heads
        self.dc_comp = config.d_comp
        self.d_c = config.d_c
        # down and up proj matrix for k and v
        self.down_proj = nn.Linear(self.dim, self.d_c)
        self.up_proj = nn.Linear(self.d_c, self.d_heads*self.num_heads)
        # down and up proj matrix for q
        self.qdown_proj = nn.Linear(self.dim, self.dc_comp)
        self.qup_proj = nn.Linear(self.dc_comp, self.d_heads*self.num_heads)
        self.w_qr = nn.Linear(self.dc_comp, self.num_heads*config.d_rope)
        self.w_kr = nn.Linear(self.dim, self.num_heads*config.d_rope)
        self.output = nn.Linear(self.num_heads*self.d_heads, self.dim)
    def forward(self, x: torch.Tensor):
        # compressed latent vecotr of keys and values
        batch_size, seq_len, _ = x.shape
        kv_compressed = self.down_proj(x)
        k_compressed = self.up_proj(kv_compressed)
        v_compressed = self.up_proj(kv_compressed)
        q_compressed = self.qdown_proj(x)
        q_c = self.qup_proj(q_compressed)
        q_r = RoPE(self.w_qr(q_compressed))
        k_r = RoPE(self.w_kr(x))
        q = torch.cat([q_c, q_r], dim=-1)
        k = torch.cat([k_compressed, k_r], dim=-1)
        scores = torch.einsum("bqhd, bkhd->bhqk", q, k) / math.sqrt(self.d_heads)
        score_soft = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk, bkhd->bqhd", score_soft, v_compressed)
        u = self.output(out.contiguous().view(batch_size, seq_len, -1)), (kv_compressed, k_r)



class RoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.d_model
        assert config.dim % 2 == 0
        self.seq_len = config.max_seq_len
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        pos = pos.view(1, x.size(1), 1, x.size(-1))
        y = torch.view_as_real(x * pos).flatten(3)
        return y


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


class MLP(nn.Module):
    def __init__(self, dim: int, i_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, i_dim)
        self.w2 = nn.Linear(dim, i_dim)
    def forward(self, x: torch.Tensor):
        x = self.w1(x)
        x = F.silu(x)
        x = self.w2(x)
        return x

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MLA(config)
        self.attn_norm = RmsNorm(config.d_model)
        self.ffn_norm = RmsNorm(config.d_model)
        self.ffn = MLP(config.d_model, config.inter_dim)
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x



