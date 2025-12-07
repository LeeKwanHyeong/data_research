# backbone.py
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = [
    "TitanBackbone",
    "MemoryEncoder",
]

from modeling_module.models.Titan.common.memory import MemoryAttention, PositionWiseFFN


class TitanBackbone(nn.Module):
    """MAC 기반 Attention → FFN 블록"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, contextual_mem_size: int, persistent_mem_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = MemoryAttention(d_model, n_heads, contextual_mem_size, persistent_mem_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class MemoryEncoder(nn.Module):
    """입력 투영 + TitanBackbone × n_layers"""

    def __init__(self, input_dim: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 contextual_mem_size: int, persistent_mem_size: int, dropout: float = 0.1,
                 use_context_update: bool = False): # <--- 인자 추가 및 기본값 False
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            TitanBackbone(d_model, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout)
            for _ in range(n_layers)
        ])
        self.use_context_update = use_context_update  # <--- 인자 값 사용

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
            # 메모리 업데이트: 각 레이어의 attn 모듈에 누적
            if self.training and self.use_context_update:
                layer.attn.update_contextual_memory(x.detach())
        return x
