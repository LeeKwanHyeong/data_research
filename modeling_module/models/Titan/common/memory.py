# memory.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PositionWiseFFN",
    "LMM",
    "MemoryAttention",
]


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LMM(nn.Module):
    """Local Memory Matching: 인코더 출력을 유사 메모리로 보강"""

    def __init__(self, d_model: int, top_k: int = 5):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k


def forward(self, encoded: torch.Tensor, memory: torch.Tensor | None) -> torch.Tensor:
    B, L, D = encoded.shape
    if memory is None or memory.numel() == 0:
        return encoded
    if memory.dim() == 2:
        memory = memory.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

    M = memory.size(1)
    k = min(self.top_k, M)
    enc_n = F.normalize(encoded, p=2, dim=-1)
    mem_n = F.normalize(memory, p=2, dim=-1)
    sim = torch.matmul(enc_n, mem_n.transpose(-2, -1))  # [B, L, M]
    _, idx = torch.topk(sim, k, dim=-1)
    mem_exp = memory.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, M, D]
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, L, k, D]
    selected = torch.gather(mem_exp, 2, idx_exp).mean(dim=2)
    return encoded + selected


class MemoryAttention(nn.Module):
    """Memory-as-Context (MAC)
    - persistent_memory: 학습 파라미터
    - contextual_memory: 최근 인코더 출력 누적(sliding)
    """

    def __init__(self, d_model: int, n_heads: int, contextual_mem_size: int, persistent_mem_size: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.persistent_memory = nn.Parameter(torch.empty(persistent_mem_size, d_model))
        nn.init.xavier_uniform_(self.persistent_memory)

        self.contextual_mem_size = contextual_mem_size
        self.register_buffer("_contextual_memory", torch.zeros(0, d_model))  # [M_c, D]

    @torch.no_grad()
    def update_contextual_memory(self, new_context: torch.Tensor):
        device = self.persistent_memory.device

        if new_context.dim() == 3:
            new_context = new_context.reshape(-1, new_context.size(-1))  # [B*L, D]
        new_context = new_context.detach().to(device)
        if new_context.numel() == 0:
            return
        mem = torch.cat([self._contextual_memory, new_context], dim=0)
        if mem.size(0) > self.contextual_mem_size:
            mem = mem[-self.contextual_mem_size:]
        self._contextual_memory = mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: [B, L, D]
        B, L, D = x.shape
        persistent = self.persistent_memory.unsqueeze(0).expand(B, -1, -1)  # [B, M_p, D]
        contextual = (self._contextual_memory.unsqueeze(0).expand(B, -1, -1)
                      if self._contextual_memory.numel() > 0
                      else torch.empty(B, 0, D, device=x.device))  # [B, M_c, D]
        mem = torch.cat([contextual, persistent, x], dim=1)  # [B, M_c+M_p+L, D]

        Q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B,h,L,d]
        K = self.W_k(mem).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [B,h,M,d]
        V = self.W_v(mem).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [B,h,M,d]

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,h,L,M]
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(ctx)
        return out
