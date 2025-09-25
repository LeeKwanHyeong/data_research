import torch
import torch.nn as nn

from models.PatchMixer.backbone import PatchMixerBackbone, MultiScalePatchMixerBackbone
from models.PatchMixer.common.configs import PatchMixerConfig
from models.PatchMixer.common.head import DecompQuantileHead


# -------------------------
# Simple PatchMixer -> Horizon regression
# -------------------------
class PatchMixerModel(nn.Module):
    """
    PatchMixerBackbone 출력(global patch represntation)
    -> MLP
    -> horizon regression
    """
    def __init__(self, configs: PatchMixerConfig):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_repr = self.backbone(x)       # (B, a * d_model)
        patch_repr = self.proj(patch_repr)  # (B, 128)
        out = self.fc(patch_repr)           # (B, H)
        return out

# -------------------------
# Simple PatchMixer + Feature Branch
# -------------------------
class PatchMixerFeatureModel(nn.Module):
    """
    Timeseries + static/exogenous feature combined head.
    feature_input: (B, feature_dim)
    """
    def __init__(self, configs: PatchMixerConfig, feature_dim: int = 4):
        super().__init__()
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

    def forward(self, ts_input: torch.Tensor, feature_input: torch.Tensor) -> torch.Tensor:
        """
        ts_input: (B, L, N)
        feature_input: (B, F)
        """
        patch_repr = self.backbone(ts_input)                              # (B, a * d_model)
        patch_repr = self.proj(patch_repr)                                # (B, 128)

        feature_repr = self.feature_mlp(feature_input)                    # (B, feature_out_dim)
        combined = torch.cat([patch_repr, feature_repr], dim = 1)  # (B, 128 + feature_out_dim)
        out = self.fc(combined)
        return out

# -------------------------
# Simple PatchMixer + Decomposition Quantile Head
# -------------------------
class PatchMixerQuantileModel(nn.Module):
    """
    Multi-Scale PatchMixer Backbone + Decomposition Quantile Head
    output: (B, 3, H) # (q10, q50, q90)
    """
    def __init__(self,
                 base_configs: PatchMixerConfig,
                 patch_cfgs = ((4, 2, 5), (8, 4, 7), (12, 6, 9)),
                 fused_dim: int = 256,
                 horizon: int | None = None,
                 per_branch_dim: int = 128,
                 fusion: str = 'concat',
                 n_harmonics: int = 4,
                 ):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, N)
        return: (B, 3, H)
        """
        z = self.backbone(x)
        q = self.head(z)
        return q


