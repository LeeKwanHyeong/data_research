import torch
import torch.nn as nn
import torch.nn.functional as F

class LMM(nn.Module):
    def __init__(self, d_model: int, top_k: int = 5):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.scale = 1.0 / (d_model ** 0.5)

    def forward(self, encoded: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        encoded: [B, L, D]
        memory : [B, M, D] or [M, D]
        return : [B, L, D]
        """
        B, L, D = encoded.shape

        # memory 존재/형상 보정
        if memory is None or memory.numel() == 0:
            return encoded  # 메모리 없으면 그대로 통과(안전)
        if memory.dim() == 2:  # [M, D] -> [B, M, D]
            memory = memory.unsqueeze(0).expand(B, -1, -1)

        M = memory.size(1)
        if M == 0:
            return encoded

        # k를 M 이내로 동적 제한
        k = min(self.top_k, M)

        # 정규화 후 유사도 계산
        encoded_norm = F.normalize(encoded, p=2, dim=-1)
        memory_norm  = F.normalize(memory,  p=2, dim=-1)
        similarity   = torch.matmul(encoded_norm, memory_norm.transpose(-2, -1))  # [B, L, M]

        # top-k 선택
        top_k_vals, top_k_idx = torch.topk(similarity, k, dim=-1)  # [B, L, k], [B, L, k]

        # 선택 메모리 gather -> 평균
        memory_exp     = memory.unsqueeze(1).expand(-1, L, -1, -1)              # [B, L, M, D]
        top_idx_exp    = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, D)          # [B, L, k, D]
        selected_mem   = torch.gather(memory_exp, 2, top_idx_exp)               # [B, L, k, D]
        matched        = selected_mem.mean(dim=2)                                # [B, L, D]

        # 간단한 residual 결합(필요시 가중치/게이팅 추가 가능)
        enhanced = encoded + matched
        return enhanced


# Efficient Multi-Head Attention with Memory
# - Titan의 MAC(Memory-as-Context) 핵심 모듈
# - 기존 Multi-Head Attention에 두 종류의 메모리 결합:
#   1) Contextual Memory → 최근 시퀀스 기반 단기 메모리 (batch별/슬라이딩 윈도우별 업데이트)
#   2) Persistent Memory → 장기 지식을 저장하는 고정 파라미터 메모리
# - Q(query)는 입력 토큰에서만 생성, K(key)/V(value)는 [Contextual + Persistent + Input] 전체에서 생성
class MemoryAttention(nn.Module):
    def __init__(self, d_model, n_heads, contextual_mem_size, persistent_mem_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        # Linear Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Persistent memory: Xavier init
        self.persistent_memory = nn.Parameter(torch.empty(persistent_mem_size, d_model))
        nn.init.xavier_uniform_(self.persistent_memory)

        # Contextual memory: dynamic buffer
        self.contextual_mem_size = contextual_mem_size
        self.contextual_memory = None

    def update_contextual_memory(self, new_context):
        device = self.persistent_memory.device
        new_context = new_context.detach().to(device)

        flat_context = new_context.view(-1, new_context.size(-1))
        if torch.isnan(flat_context).any():
            print("[Warning] Skipping update: NaN in new_context")
            return

        flat_context = flat_context.clamp(min=-1e3, max=1e3)

        if self.contextual_memory is None:
            self.contextual_memory = flat_context
        else:
            self.contextual_memory = torch.cat([self.contextual_memory, flat_context], dim=0)
            if self.contextual_memory.size(0) > self.contextual_mem_size:
                self.contextual_memory = self.contextual_memory[-self.contextual_mem_size:]

    def forward(self, x):
        B, L, _ = x.size()

        # Memory Construction
        persistent_mem = self.persistent_memory.unsqueeze(0).expand(B, -1, -1)
        contextual_mem = (
            torch.empty(B, 0, self.d_model, device=x.device)
            if self.contextual_memory is None
            else self.contextual_memory.unsqueeze(0).expand(B, -1, -1)
        )
        extended_input = torch.cat([contextual_mem, persistent_mem, x], dim=1)

        # Q: from input, K/V: from extended memory
        Q = self.W_q(x)
        K = self.W_k(extended_input)
        V = self.W_v(extended_input)

        # Reshape for Multi-head
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, K.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, V.size(1), self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9, posinf=1e4, neginf=-1e4)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

        context = torch.matmul(attn_probs, V)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.out_proj(context)

        return output


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))