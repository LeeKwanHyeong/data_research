from typing import Sequence, Optional

import torch
from torch import Tensor
from torch import nn

from models.PatchTST.common.configs import PatchTSTConfig
from models.PatchTST.common.heads.delta_quantile_head import DeltaQuantileHead
from models.PatchTST.common.heads.monotone_quantile_head import MonotoneQuantileHead
from models.PatchTST.supervised.backbone import SupervisedBackbone
from models.components.decomposition import CausalMovingAverage1D


class BaseModel(nn.Module):
    """
    입력:  x (B, L, C)  - Batch, Lookback, Channels
    출력:  ŷ (B, H, C)  - Batch, Horizon, Channels (외부 파이프라인 정합을 위해 통일)
    내부: PatchTST Backbone은 (B, L, C) -> (B, n_vars, H) 형식으로 동작
    """
    def __init__(self, configs: PatchTSTConfig):
        super().__init__()
        self.configs = configs
        self.model_name = 'PatchTST BaseModel'

        # ----- PatchTST Backbone -----
        # 백본은 (B, C, L) 입력을 받아 Flatten_Head로 (B, nvars, H) 반환
        self.backbone = SupervisedBackbone(self.configs)

        # ----- Decomposition -----
        # configs.decomp가 없거나 비활성화면 None 처리
        self.decomp_type = getattr(getattr(configs, 'decomp', None), 'type', None)
        self.decomp_type = (self.decomp_type or 'none').lower()
        if self.decomp_type == 'ma':
            self.decomp = CausalMovingAverage1D(window=configs.decomp.ma_window)
        else:
            self.decomp = None

        # 'concat' | 'residual_only' | 'trend_only'
        self.concat_mode = getattr(getattr(configs, 'decomp', None), 'concat_mode', 'residual_only')
        if self.decomp is None:
            self.concat_mode = 'residual_only'  # 비활성 시 내부적으로 무시

        # concat이면 입력 채널이 2C가 되므로 1x1 Conv로 2C→C 매핑 (백본 c_in과 정합 유지)
        c_in = configs.c_in
        if self.decomp and self.concat_mode == 'concat':
            self.input_mapper = nn.Conv1d(in_channels=2 * c_in, out_channels=c_in, kernel_size=1, bias=False)
        else:
            self.input_mapper = nn.Identity()

        # ----- Future EXO ------
        # 우선 config.exo_dim을 우선, 인자로 덮어쓰기
        self.exo_dim = int(getattr(configs, 'exo_dim', 0))

        self.horizon = configs.horizon

        # exo_dim > 0이면 (B, H, exo_dim) -> (B, H, 1) time-distributed 가산항
        self.exo_head: Optional[nn.Module] = None
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)    # (B, H, exo_dim) -> (B, H, 1)
            )

    # ----- Inner Utility -----
    def _apply_decomposition(self, x_bcl: Tensor) -> Tensor:
        """x_bcl: (B, C, L) -> (B, C or 2C, L)"""
        if not self.decomp:
            return x_bcl
        trend, residual = self.decomp(x_bcl)  # (B, C, L), (B, C, L)
        if self.concat_mode == 'concat':
            x_bcl = torch.cat([trend, residual], dim=1)  # (B, 2C, L)
        elif self.concat_mode == 'trend_only':
            x_bcl = trend
        else:
            # 'residual_only'
            x_bcl = residual
        return x_bcl

    def _align_exo_len(self, exo: Tensor, Hm: int, device, dtype) -> Tensor:
        """
        exo: (B, H, exo_dim) -> (B, Hm, exo_dim)
        """
        B, Hx, E = exo.size()
        exo = exo.to(device = device, dtype = dtype)
        if Hx == Hm:
            return exo
        if Hx > Hm:
            return exo[:, :Hm, :]

        pad = torch.zeros(B, Hm -Hx, E, device = device, dtype = dtype)
        return torch.cat([exo, pad], dim = 1)

    def forward(self, x: Tensor, future_exo: Tensor | None = None) -> Tensor:
        """
        x: (B, L, C)  -> permute -> (B, C, L) -> [decomp] -> input_mapper -> backbone
        backbone 출력: (B, nvars, H) -> permute -> (B, H, nvars)
        (optional) exogenous addition: (B, H, 1) broadcast-add
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D [B, L, C], gott {tuple(x.shape)}")

        # 1) 입력 차원 정렬 (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)  # (B, C, L)

        # 2) decomposition (optional)
        if self.decomp:
            x = self._apply_decomposition(x)

        # 3) 채널 정합 (concat일 때 2C→C)
        x = self.input_mapper(x)

        # 4) 백본 통과
        pred = self.backbone(x)  # (B, nvars, H)

        # 5) 출력 차원 정규화
        pred = pred.permute(0, 2, 1).contiguous()  # (B, H, nvars)

        # 6) EXO 가산 (옵션)
        if (self.exo_head is not None) and (future_exo is not None):
            pred = pred.contiguous()
            B, Hm, C = pred.shape

            # device/dtype 정렬
            future_exo = self._align_exo_len(future_exo, Hm, device = pred.device, dtype = pred.dtype)
            ex = self.exo_head(future_exo).squeeze(-1)  # (B, H)
            pred = pred + ex.unsqueeze(-1)              # (B, H, C)

        return pred

class QuantileModel(BaseModel):
    """
    BaseModel의 Point forecasting 기반으로, 시간 축(H) 위치별로
    (Point Scalar + EXO(optional) -> Quantile Vector(Q)로 Projection하는 얕은 Head.
    return: (B, Q, H) (target_channel에 대한 분위수들)
    """
    def __init__(
            self,
            configs,
            quantiles: Sequence[float] = (0.1, 0.5, 0.9),
            target_channel: int = 0,
            q_hidden: int = 64,
            monotonic: bool = True,
            use_exo_in_head: bool = False,
    ):
        # 반드시 BaseModel을 상속하고 super().__init__호출
        super().__init__(configs)
        self.is_quantile = True
        self.quantiles = tuple(float(q) for q in quantiles)
        assert len(self.quantiles) == 3, "이 구현은 q10,q50,q90(3분위) 기준입니다."
        self.target_channel = int(target_channel)
        self.monotonic = bool(monotonic)
        self.use_exo_in_head = bool(use_exo_in_head)

        # ---- in_features: d_model (+ point + exo) ----
        d_model = configs.d_model
        add_point = 1  # point(중앙값) 특성 1차원 추가 (권장)
        exo_dim = int(getattr(configs, 'exo_dim', 0)) if use_exo_in_head else 0
        self._feat_dim = d_model + add_point + exo_dim

        self.delta_head = DeltaQuantileHead(
            in_features=self._feat_dim,
            hidden=q_hidden,
            dropout=configs.head.head_dropout
        )

    def forward(self, x, future_exo=None):
        # 1) 부모 포인트 실행
        pred_bhc = super().forward(x, future_exo)  # [B, H, C]
        B, Hm, C = pred_bhc.shape
        if C == 1:
            pred_bh = pred_bhc.squeeze(-1)
        else:
            if not (0 <= self.target_channel < C):
                raise IndexError(f"target_channel={self.target_channel} out of range [0,{C - 1}]")
            pred_bh = pred_bhc[..., self.target_channel]  # [B, H]

        # 2) 히든 컨텍스트 요약(평균) → [B,H,d_model] 로 확장
        z = getattr(self.backbone, "_last_z", None)  # [B,nvars,d_model,N]
        if z is None:
            raise RuntimeError("backbone must cache _last_z before head.")
        z_t = z[:, self.target_channel]  # [B,d_model,N]
        z_ctx = z_t.mean(dim=2)  # [B,d_model]
        z_flat = z_ctx.unsqueeze(1).expand(B, Hm, -1)  # [B,H,d_model]

        # 3) point/EXO 붙여 feats 구성
        feats = torch.cat([z_flat, pred_bh.unsqueeze(-1)], dim=-1)  # +1
        if self.use_exo_in_head and (future_exo is not None) and getattr(self.configs, 'exo_dim', 0) > 0:
            exo = self._align_exo_len(future_exo, Hm, device=feats.device, dtype=feats.dtype)  # [B,H,E]
            feats = torch.cat([feats, exo], dim=-1)

        # 4) Δ 예측 + 최소 폭(eps) 보정
        deltas = self.delta_head(feats)  # [B,H,2]  (d_minus, d_plus)
        # eps: point 규모 기반의 작은 바닥치(시각화 안정 목적)
        # |point|의 1% + 1e-3 (원한다면 0.5~5%로 조절)
        eps = (pred_bh.abs().unsqueeze(-1) * 0.01) + 1e-3  # [B,H,1]

        m = pred_bh.unsqueeze(-1)  # [B,H,1]
        d_minus = deltas[..., 0:1] + eps  # [B,H,1]
        d_plus = deltas[..., 1:1 + 1] + eps  # [B,H,1]

        q10 = m - d_minus
        q50 = m
        q90 = m + d_plus

        q = torch.cat([q10, q50, q90], dim=-1)  # [B,H,3]
        return q.permute(0, 2, 1).contiguous()  # [B,3,H]