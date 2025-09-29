import torch
from torch import nn

class MonotoneQuantileHead(nn.Module):
    """
    입력: z_flat [B, H, F] (F = d_model * patch_num (+ exo_dim optional))
    출력: [B, H, 3] (q10, q50, q90), 단조성 보장
    """
    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mu = nn.Linear(hidden, 1)                                  # median
        self.d1 = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())    # >= 0
        self.d2 = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())    # >= 0

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        h = self.net(z_flat)   # [B, H, Hidden]
        m = self.mu(h)         # [B, H, 1]
        d1 = self.d1(h)        # [B, H, 1]
        d2 = self.d2(h)        # [B, H, 1]
        q10 = m - d1
        q50 = m
        q90 = m + d2
        return torch.cat([q10, q50, q90], dim = -1) # [B, H, 3]

