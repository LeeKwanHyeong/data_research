# decoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TitanDecoderLayer",
    "TitanDecoder",
]


class TitanDecoderLayer(nn.Module):
    """
    Causal Self-Attn + Cross-Attn + FFN 구조의 디코더 레이어
    - 입력: tgt [B, H, D], memory [B, L, D]
    - 출력: [B, H, D]
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal

    @staticmethod
    def _causal_mask(H: int, device: torch.device) -> torch.Tensor:
        # 상삼각 마스크(미래 차단)
        m = torch.full((H, H), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # 1) causal self-attn
        q = self.norm1(tgt)
        attn_mask = self._causal_mask(q.size(1), q.device) if self.causal else None
        x, _ = self.self_attn(q, q, q, attn_mask=attn_mask)
        x = tgt + self.dropout(x)  # residual

        # 2) cross-attn
        y = self.norm2(x)
        z, _ = self.cross_attn(y, memory, memory)
        x = x + self.dropout(z)

        # 3) ffn
        w = self.norm3(x)
        f = self.ffn(w)
        out = x + f
        return out


# decoder.py
class TitanDecoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 horizon: int = 60,
                 exo_dim: int = 0,
                 causal: bool = True):
        super().__init__()
        self.horizon = horizon
        self.exo_dim = exo_dim

        # 기존 임베딩
        self.query_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # [수정 1] 여기에 이 줄을 꼭 추가해야 합니다!
        self.enc_last_proj = nn.Linear(d_model, d_model)

        self.exo_proj = nn.Linear(exo_dim, d_model) if exo_dim > 0 else None
        self.layers = nn.ModuleList([
            TitanDecoderLayer(d_model, n_heads, d_ff, dropout, causal)
            for _ in range(n_layers)
        ])

    def forward(self, memory: torch.Tensor, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        # memory: [B, L, D]
        B, L, D = memory.shape
        H = self.horizon

        # [수정 2] forward 로직 (작성하신 대로 유지)
        enc_last = memory[:, -1:, :]  # [B, 1, D]

        # __init__에 정의가 되어 있어야 아래 줄이 실행됩니다.
        enc_last_emb = self.enc_last_proj(enc_last).expand(B, H, D)

        tgt = self.query_embed.expand(B, H, D) + self.pos_embed.expand(B, H, D) + enc_last_emb

        if (self.exo_proj is not None) and (future_exo is not None):
            tgt = tgt + self.exo_proj(future_exo)  # [B, H, D]

        x = tgt
        for layer in self.layers:
            x = layer(x, memory)
        return x

# class TitanDecoder(nn.Module):
#     """
#     Titan용 Causal Transformer 디코더 스택
#     - query_embed + pos_embed를 초기 입력으로 사용
#     - (선택) 미래 exogenous를 proj하여 입력에 더함
#     """
#
#     def __init__(self,
#                  d_model: int,
#                  n_layers: int = 2,
#                  n_heads: int = 4,
#                  d_ff: int = 512,
#                  dropout: float = 0.1,
#                  horizon: int = 60,
#                  exo_dim: int = 0,
#                  causal: bool = True):
#         super().__init__()
#         self.horizon = horizon
#         self.exo_dim = exo_dim
#         self.query_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
#         self.pos_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
#         self.exo_proj = nn.Linear(exo_dim, d_model) if exo_dim > 0 else None
#         self.layers = nn.ModuleList([
#             TitanDecoderLayer(d_model, n_heads, d_ff, dropout, causal)
#             for _ in range(n_layers)
#         ])
#
#     def forward(self, memory: torch.Tensor, future_exo: torch.Tensor | None = None) -> torch.Tensor:
#         # memory: [B, L, D]
#         B, L, D = memory.shape
#         H = self.horizon
#
#         # #
#         enc_last = memory[:, -1:, :]
#         enc_last_emb = self.enc_last_proj(enc_last).expand(B, H, D)
#         tgt = self.query_embed.expand(B, H, D) + self.pos_embed.expand(B, H, D) + enc_last_emb
#         # tgt = self.query_embed.expand(B, H, D) + self.pos_embed.expand(B, H, D)
#         if (self.exo_proj is not None) and (future_exo is not None):
#             tgt = tgt + self.exo_proj(future_exo)  # [B, H, D]
#
#         x = tgt
#         for layer in self.layers:
#             x = layer(x, memory)
#         return x  # [B, H, D]
