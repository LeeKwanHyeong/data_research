import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Causal Decoder Layer ---
class TitanDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1, causal = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim = d_model, num_heads = n_heads, dropout = dropout, batch_first = True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim = d_model, num_heads = n_heads, dropout = dropout, batch_first = True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal

    def _causal_mask(self, H, device):
        # 상삼각(미래 차단)
        m = torch.full((H, H), float("-inf"), device = device)
        return torch.triu(m, diagonal = 1)

    def forward(self, tgt, memory):
        # tgt: [B, H, D], memory: [B, L, D]
        q = self.norm1(tgt)
        attn_mask = self._causal_mask(q.size(1), q.device) if self.causal else None
        x, _ = self.self_attn(q, q, q, attn_mask = attn_mask)   # masked self-attn
        x = tgt + self.dropout(x)

        y = self.norm2(x)
        z, _ = self.cross_attn(y, memory, memory)   # cross-attn
        x = x + self.dropout(z)

        w = self.norm3(x)
        f = self.ffn(w)
        out = x + f
        return out

# --- Causal Decoder Stack ---
class TitanDecoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_layers: int = 1,
                 n_heads: int = 4,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 horizon: int = 60,
                 exo_dim: int = 0,
                 causal: bool = True):
        super().__init__()
        self.horizon = horizon
        self.exo_dim = exo_dim

        self.query_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        self.exo_proj = nn.Linear(exo_dim, d_model) if exo_dim > 0 else None

        self.layers = nn.ModuleList([
            TitanDecoderLayer(d_model, n_heads, d_ff, dropout, causal)
            for _ in range(n_layers)
        ])

    def forward(self, memory: torch.Tensor, future_exo: torch.Tensor | None = None):
        """
        memory:     [B, L, D] (encoder output)
        future_exo: [B, H, exo_dim] or None
        return:     [B, H, D]
        """
        B, L, D = memory.shape
        H = self.horizon

        # future query(learned) + pos emb ( + exo)
        tgt = self.query_embed.expand(B, H, D) + self.pos_embed.expand(B, H, D)
        if (self.exo_proj is not None) and (future_exo is not None):
            tgt = tgt + self.exo_proj(future_exo)

        x = tgt

        for layer in self.layers:
            x = layer(x, memory)
        return x # [B, H, D]

