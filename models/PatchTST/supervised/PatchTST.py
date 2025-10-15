import torch
from torch import nn

from models.PatchTST.supervised.backbone import SupervisedBackbone


# ------------------------------
# 1) Point Forecast Head
# ------------------------------
class PointHead(nn.Module):
    """
    단일 포인트 예측 헤드.
    Backbone에서 나온 feature sequence (B, L_tok, d_model)을 받아서
    평균 혹은 마지막 토큰을 이용해 (B, horizon) 출력으로 변환.
    """
    def __init__(self, d_model: int, horizon: int, agg: str = "mean"):
        super().__init__()
        self.agg = agg
        self.proj = nn.Linear(d_model, horizon)

    def forward(self, z_bld: torch.Tensor) -> torch.Tensor:
        # z_bld: [B, L_tok, d_model]
        feat = z_bld.mean(dim=1) if self.agg == "mean" else z_bld[:, -1, :]
        return self.proj(feat)  # [B, horizon]


# ------------------------------
# 2) Quantile Forecast Head
# ------------------------------
class QuantileHead(nn.Module):
    """
    다중 분위수 예측 헤드.
    (B, L_tok, d_model) -> (B, Q, horizon)
    """
    def __init__(self,
                 d_model: int,
                 horizon: int,
                 quantiles=(0.1, 0.5, 0.9),
                 hidden: int = 128,
                 monotonic: bool = True):
        super().__init__()
        self.Q = len(quantiles)
        self.horizon = horizon
        self.monotonic = monotonic

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * self.Q)
        )

    def forward(self, z_bld: torch.Tensor) -> torch.Tensor:
        B = z_bld.size(0)
        feat = z_bld.mean(dim=1)  # [B, d_model]
        q = self.net(feat).view(B, self.horizon, self.Q)  # [B, H, Q]
        q = q.permute(0, 2, 1).contiguous()               # [B, Q, H]
        if self.monotonic:
            q, _ = torch.sort(q, dim=1)
        return q


# ------------------------------
# 3) Full Models
# ------------------------------
class PatchTSTPointModel(nn.Module):
    """
    PatchTST Backbone + Point Head
    """
    def __init__(self, cfg, attn_core=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = SupervisedBackbone(cfg, attn_core)
        self.head = PointHead(cfg.d_model, cfg.horizon)
        self.is_quantile = False
        self.horizon = cfg.horizon
        self.model_name = "PatchTST BaseModel"

    def forward(self, x_b_l_c: torch.Tensor) -> torch.Tensor:
        """
        x_b_l_c: [B, L, C]
        return: [B, horizon]
        """
        x = x_b_l_c.permute(0, 2, 1)  # [B, C, L]
        z = self.backbone(x)          # [B, L_tok, d_model]
        y = self.head(z)              # [B, horizon]
        return y


class PatchTSTQuantileModel(nn.Module):
    """
    PatchTST Backbone + Quantile Head
    """
    def __init__(self, cfg, attn_core=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = SupervisedBackbone(cfg, attn_core)
        self.head = QuantileHead(cfg.d_model, cfg.horizon,
                                 quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)))
        self.is_quantile = True
        self.horizon = cfg.horizon
        self.model_name = "PatchTST QuantileModel"

    def forward(self, x_b_l_c: torch.Tensor) -> torch.Tensor:
        """
        x_b_l_c: [B, L, C]
        return: [B, Q, horizon]
        """
        x = x_b_l_c.permute(0, 2, 1)  # [B, C, L]
        z = self.backbone(x)          # [B, L_tok, d_model]
        q = self.head(z)              # [B, Q, horizon]
        return q