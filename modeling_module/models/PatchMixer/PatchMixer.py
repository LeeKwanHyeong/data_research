import torch
import torch.nn as nn

from modeling_module.models.PatchMixer.backbone import PatchMixerBackbone, MultiScalePatchMixerBackbone
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchMixer.common.head import DecompQuantileHead
from modeling_module.utils.exogenous_utils import _apply_exo_shift_linear


# -------------------------
# Simple PatchMixer -> Horizon regression
# -------------------------
class BaseModel(nn.Module):
    """
    PatchMixerBackbone 출력(global patch represntation)
    -> MLP
    -> horizon regression
    """
    def __init__(self, configs: PatchMixerConfig, exo_dim: int = 0):
        super().__init__()
        self.model_name = 'PatchMixer BaseModel'

        self.backbone = PatchMixerBackbone(configs = configs)

        in_dim = self.backbone.patch_repr_dim # a * d_model
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Softplus(),
            nn.Dropout(0.10)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, configs.horizon)
        )

        self.horizon = configs.horizon
        self.exo_dim = exo_dim
        self.exo_head = None
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

    def forward(self, x: torch.Tensor, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        """
                x: (B, L, N)
                future_exo: (B, Hx, exo_dim) | None
                mode/kwargs: 무시(호출자 호환성)
                """
        patch_repr = self.backbone(x)  # (B, a * d_model)
        patch_repr = self.proj(patch_repr)  # (B, 128)
        out = self.fc(patch_repr)  # (B, H)

        if (self.exo_head is not None) and (future_exo is not None):
            ex = _apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon=self.horizon,
                out_dtype=out.dtype,
                out_device=out.device
            )  # (B, H)
            out = out + ex

        return out  # (B, H)

# -------------------------
# Simple PatchMixer + Feature Branch
# -------------------------
class FeatureModel(nn.Module):
    """
    Timeseries + static/exogenous feature combined head.
    ts_input:      (B, L, N)
    feature_input: (B, F)
    future_exo:    (B, Hx, exo_dim) | None
    """
    def __init__(self, configs: PatchMixerConfig, feature_dim: int = 4, exo_dim: int = 0):
        super().__init__()
        self.model_name = 'PatchMixer FeatureModel'

        self.backbone = PatchMixerBackbone(configs = configs)

        # Feature Branch
        self.feature_out_dim = 8
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, self.feature_out_dim),
            nn.ReLU()
        )

        # Patch representation projection
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.patch_repr_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10)
        )

        combined_dim = 128 + self.feature_out_dim

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, configs.horizon)
        )

        self.horizon = int(configs.horizon)
        self.exo_dim = int(exo_dim)
        self.exo_head = None
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),  # (B, Hx, 1)
            )

    def forward(self,
                ts_input: torch.Tensor,
                feature_input: torch.Tensor,
                future_exo: torch.Tensor | None = None
                ) -> torch.Tensor:
        """
        ts_input: (B, L, N)
        feature_input: (B, F)
        """
        patch_repr = self.backbone(ts_input)                              # (B, a * d_model)
        patch_repr = self.proj(patch_repr)                                # (B, 128)

        feature_repr = self.feature_mlp(feature_input)                    # (B, feature_out_dim)
        combined = torch.cat([patch_repr, feature_repr], dim = 1)  # (B, 128 + feature_out_dim)
        out = self.fc(combined)

        if (self.exo_head is not None) and (future_exo is not None):
            ex = _apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon=self.horizon,
                out_dtype=out.dtype,
                out_device=out.device
            )  # (B, H)
            out = out + ex

        return out  # (B, H)
# -------------------------
# Simple PatchMixer + Decomposition Quantile Head
# -------------------------
class QuantileModel(nn.Module):
    """
    Multi-Scale PatchMixer Backbone + Decomposition Quantile Head
    output: (B, 3, H)  # (q10, q50, q90)
    + (선택) exogenous shift를 모든 quantile에 동일 적용
    """
    def __init__(self,
                 base_configs: PatchMixerConfig,
                 patch_cfgs = ((4, 2, 5), (8, 4, 7), (12, 6, 9)),
                 fused_dim: int = 256,
                 horizon: int | None = None,
                 per_branch_dim: int = 128,
                 fusion: str = 'concat',
                 n_harmonics: int = 4,
                 exo_dim: int = 0
                 ):
        super().__init__()
        self.is_quantile = True
        self.model_name = 'PatchMixer QuantileModel'

        H = horizon if horizon is not None else base_configs.horizon

        self.backbone = MultiScalePatchMixerBackbone(
            base_configs=base_configs,
            patch_cfgs=patch_cfgs,
            per_branch_dim=per_branch_dim,
            fused_dim=fused_dim,
            fusion=fusion,
        )

        self.head = DecompQuantileHead(
            in_dim=self.backbone.out_dim,
            horizon=H,
            quantiles=(0.1, 0.5, 0.9),
            n_harmonics=n_harmonics,
        )
        self.horizon = H
        self.exo_dim = exo_dim
        self.exo_head = None
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

    def forward(self, x: torch.Tensor, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, L, N)
        return: (B, 3, H)
        """
        z = self.backbone(x)    # (B, fused_dim)
        q = self.head(z)        # (B, 3, H)

        if (self.exo_head is not None) and (future_exo is not None):
            ex = _apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon=self.horizon,
                out_dtype=q.dtype,
                out_device=q.device
            )  # (B, H)
            q = q + ex.unsqueeze(1)  # (B, 1, H) → 모든 분위수에 동일 shift

        return q  # (B, 3, H)

