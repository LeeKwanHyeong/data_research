import torch
import torch.nn as nn

from modeling_module.models.PatchMixer.backbone import PatchMixerBackbone, MultiScalePatchMixerBackbone
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.common_layers.RevIN import RevIN
from modeling_module.models.common_layers.heads.quantile_heads.decomposition_quantile_head import \
    DecompositionQuantileHead
from modeling_module.utils.exogenous_utils import _apply_exo_shift_linear
from modeling_module.utils.temporal_expander import TemporalExpander


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
        self.revin = RevIN(configs.enc_in)

    def forward(self, x: torch.Tensor, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        """
                x: (B, L, N)
                future_exo: (B, Hx, exo_dim) | None
                mode/kwargs: 무시(호출자 호환성)
                """
        x = self.revin(x, 'norm')
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
        out = self.revin(out, 'denorm')

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
        self.revin = RevIN(configs.enc_in)

    def forward(self,
                ts_input: torch.Tensor,
                feature_input: torch.Tensor,
                future_exo: torch.Tensor | None = None
                ) -> torch.Tensor:
        """
        ts_input: (B, L, N)
        feature_input: (B, F)
        """
        ts_input = self.revin(ts_input, 'norm')
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
        out = self.revin(out, 'denorm')
        return out  # (B, H)
# -------------------------
# Simple PatchMixer + Decomposition Quantile Head
# -------------------------
class QuantileModel(nn.Module):
    """
    Multi-Scale PatchMixer Backbone + DecompQuantileHeadV2
    output: (B, 3, H)  # (q10, q50, q90)
    + (선택) exogenous shift 동일 적용
    """
    def __init__(self,
                 base_configs,
                 patch_cfgs = ((4, 2, 5), (8, 4, 7), (12, 6, 9)),
                 fused_dim: int = 256,
                 horizon: int | None = None,
                 per_branch_dim: int = 128,
                 fusion: str = 'concat',
                 n_harmonics: int = 4,
                 exo_dim: int = 0,
                 f_out: int = 128 # <-- expander 출력 차원
                 ):
        super().__init__()
        self.is_quantile = True
        self.model_name = 'PatchMixer QuantileModel'

        H = horizon if horizon is not None else base_configs.horizon
        self.horizon = int(H)

        # 1) Backbone: 전역 벡터 [B, D]
        self.backbone = MultiScalePatchMixerBackbone(
            base_configs=base_configs,
            patch_cfgs=patch_cfgs,
            per_branch_dim=per_branch_dim,
            fused_dim=fused_dim,
            fusion=fusion,
        )
        d_in = self.backbone.out_dim

        # 2) Temporal Expander: [B,D] -> [B,H,F]
        self.expander = TemporalExpander(d_in=d_in, horizon=H, f_out=f_out, dropout=0.1)

        # 3) Decomposition Quantile Head (V2): [B,H,F] -> [B,Q,H]
        self.head = DecompositionQuantileHead(
            in_features=f_out,
            quantiles=[0.1, 0.5, 0.9],
            hidden=128,
            dropout=float(getattr(base_configs, 'head_dropout', 0.0) or 0.0),
            mid=0.5,
            use_trend=True,
            fourier_k=n_harmonics,
            agg="mean",
        )

        # (선택) exogenous shift
        self.exo_dim = int(exo_dim)
        self.exo_head = None
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

        self.revin = RevIN(base_configs.enc_in)  # enc_in=1 가정

    def _denorm_quantiles_with_revin(self, q_bqh: torch.Tensor) -> torch.Tensor:
        """
        q_bqh: [B,Q,H]  (Q=3)
        RevIN은 [B,L,N] 형태(또는 [B,L] 변형)에 맞춰 denorm해야 하므로,
        분위수별로 [B,H,1]로 만들어 각각 denorm 적용 후 다시 합칩니다.
        """
        B, Q, H = q_bqh.shape
        outs = []
        for i in range(Q):
            yi = q_bqh[:, i, :]            # [B,H]
            yi = yi.unsqueeze(-1)          # [B,H,1]  (N=1)
            yi = self.revin(yi, 'denorm')  # RevIN이 [B,L,N]을 받는다고 가정
            outs.append(yi.squeeze(-1))    # [B,H]
        return torch.stack(outs, dim=1)     # [B,Q,H]

    def _ensure_bqh(self, q: torch.Tensor, horizon: int, qlen: int) -> torch.Tensor:
        # 허용: (B,Q,H) 또는 (B,H,Q)
        if q.dim() != 3:
            raise ValueError(f"pred must be 3D, got {q.shape}")
        B, A, Bdim = q.shape
        if A == qlen and Bdim == horizon:  # (B,Q,H)
            return q
        if A == horizon and Bdim == qlen:  # (B,H,Q)
            return q.permute(0, 2, 1).contiguous()
        raise ValueError(f"pred shape must be (B,{qlen},{horizon}) or (B,{horizon},{qlen}), got {q.shape}")

    def forward(self, x: torch.Tensor, future_exo: torch.Tensor | None = None, *,
                exo_is_normalized: bool = True) -> torch.Tensor:
        """
        x: (B, L, N)  # RevIN이 이 형태를 받는 구현 가정
        return: (B, 3, H)
        """
        # 0) 입력 정규화
        x_n = self.revin(x, 'norm')

        # 1) 백본: 전역 벡터
        z = self.backbone(x_n)               # (B, D)

        # 2) 시점 확장
        x_bhf = self.expander(z)             # (B, H, F)

        # 3) 분위수 예측(교차 방지 포함)
        q = self.head(x_bhf)                 # (B, 3, H)  normalized space

        q = self._ensure_bqh(q, self.horizon, qlen=3)

        # 4) (선택) exogenous shift
        #    - exo_is_normalized=True: RevIN 기준 공간에서 학습/입력된 exo라면 denorm 이전에 더함
        #    - exo_is_normalized=False: 원 단위라면 denorm 이후에 더해야 함
        if (self.exo_head is not None) and (future_exo is not None) and exo_is_normalized:
            ex = _apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon=self.horizon,
                out_dtype=q.dtype,
                out_device=q.device
            )  # (B, H)
            q = q + ex.unsqueeze(1)          # (B, 3, H)


        # 6) exogenous가 원 단위일 때는 여기서 더하세요.
        if (self.exo_head is not None) and (future_exo is not None) and (not exo_is_normalized):
            ex = _apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon=self.horizon,
                out_dtype=q.dtype,
                out_device=q.device
            )  # (B, H)
            q = q + ex.unsqueeze(1)          # (B, 3, H)

        # 5) RevIN 역정규화 (분위수별 슬라이스)
        q = self._denorm_quantiles_with_revin(q)  # (B, 3, H)

        return q

