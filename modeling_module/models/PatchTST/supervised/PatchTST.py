import torch
from torch import nn

from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.PatchTST.supervised.backbone import SupervisedBackbone
from modeling_module.models.common_layers.RevIN import RevIN



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

        # Forecaster 호환: 속성명을 revin_layer로 통일
        self.revin_layer = RevIN(num_features=cfg.c_in)

    @classmethod
    def from_config(cls, config: "PatchTSTConfig"):
        return cls(cfg=config)

    def forward(self, x_b_l_c: torch.Tensor, future_exo=None, mode=None) -> torch.Tensor:
        """
        x_b_l_c: [B, L, C] (원공간)
        반환: y_n: [B, H] (정규화 공간; Forecaster에서 model.revin_layer(y, 'denorm') 처리)
        """
        # 1) 입력 정규화 (컨텍스트: 배치별/채널별 통계가 revin_layer 내부에 저장)
        x_n = self.revin_layer(x_b_l_c, 'norm')          # [B, L, C]

        # 2) Backbone forward (PatchTST는 [B, C, L] 입력)
        x_n_b_c_l = x_n.permute(0, 2, 1)                 # [B, C, L]
        z = self.backbone(x_n_b_c_l)                     # [B, L_tok, d_model]

        # 3) Head → normalized space 예측
        y_n = self.head(z)                               # [B, H]
        return y_n                                       # denorm은 Forecaster가 일원화 수행

class PatchTSTQuantileModel(nn.Module):
    """
    PatchTST Backbone + Quantile Head (quantile forecast)
    - 입력은 RevIN으로 'norm'만 수행
    - 출력은 normalized space 그대로 반환(dict with 'q')
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

        # Forecaster 호환: 속성명을 revin_layer로 통일
        self.revin_layer = RevIN(num_features=cfg.c_in)

    @classmethod
    def from_config(cls, config: "PatchTSTConfig"):
        return cls(cfg=config)

    def forward(self, x_b_l_c: torch.Tensor, future_exo=None, mode=None):
        """
        x_b_l_c: [B, L, C] (원공간)
        반환: {"q": q_n}  with q_n: [B, H, Q] (정규화 공간)
              (Forecaster가 'q'를 우선 선택해 horizon을 정규화하고, revin_layer로 denorm)
        """
        # 1) 입력 정규화
        x_n = self.revin_layer(x_b_l_c, 'norm')          # [B, L, C]

        # 2) Backbone forward (PatchTST는 [B, C, L] 입력)
        x_n_b_c_l = x_n.permute(0, 2, 1)                 # [B, C, L]
        z = self.backbone(x_n_b_c_l)                     # [B, L_tok, d_model]

        # 3) Head → normalized space quantiles
        q_n = self.head(z)                               # [B, H, Q]
        return {"q": q_n}