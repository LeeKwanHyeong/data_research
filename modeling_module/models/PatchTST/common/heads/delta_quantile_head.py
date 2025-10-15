import torch
from torch import nn

class DeltaQuantileHead(nn.Module):
    """
    입력: feats [B, H, F]
    출력: deltas [B, H, 2]  (d_minus, d_plus), 둘 다 >= 0 (softplus)
    q50는 외부에서 주입(= point), 최종 q는:
      q10 = q50 - (d_minus + eps)
      q50 = q50
      q90 = q50 + (d_plus  + eps)
    """
    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.sp = nn.Softplus()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, H, F]
        d = self.net(feats)                 # [B, H, 2]
        return self.sp(d)                   # >= 0
