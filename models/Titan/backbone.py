import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Titan.common.memory import MemoryAttention, PositionWiseFFN


def debug_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        print(f"[{name}] Not a tensor")
        return
    print(f"[{name}] shape: {tensor.shape}")
    print(f"  - has_nan: {torch.isnan(tensor).any().item()}, has_inf: {torch.isinf(tensor).any().item()}")
    print(f"  - max: {tensor.max().item():.4f}, min: {tensor.min().item():.4f}")


class TitanBackbone(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout = 0.1):
        super().__init__()
        self.attn = MemoryAttention(d_model, n_heads, contextual_mem_size, persistent_mem_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout = dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# Memory Encoder (Core Network)
class MemoryEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            TitanBackbone(d_model, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x
