import torch.nn as nn

# models/common_layers/TrendCorrector.py

import torch
from torch import nn, Tensor

class TrendCorrector(nn.Module):
    """
    마지막 토큰(또는 2D 입력)을 받아 out_dim로 보정치를 내는 작은 헤드.
    - (B,L,D) → 마지막 토큰을 뽑아 (B,D)
    - (B,D)   → 그대로 처리
    구성: [LayerNorm] → Linear(D→D) → GELU → Dropout → Linear(D→out_dim)
    """
    def __init__(self, d_model: int, out_dim: int, *, use_ln: bool = True,
                 hidden_mult: float = 1.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(1, int(d_model * hidden_mult))

        self.use_ln = use_ln
        self.norm = nn.LayerNorm(d_model) if use_ln else nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, encoded_seq: Tensor) -> Tensor:
        """
        encoded_seq: (B,L,D) or (B,D)
        returns: (B, out_dim)
        """
        if encoded_seq.dim() == 3:
            x = encoded_seq[:, -1, :]   # (B, D)
        elif encoded_seq.dim() == 2:
            x = encoded_seq             # (B, D)
        else:
            raise RuntimeError(f"TrendCorrector expects (B,L,D) or (B,D), got {tuple(encoded_seq.shape)}")

        x = self.norm(x)
        return self.mlp(x)
