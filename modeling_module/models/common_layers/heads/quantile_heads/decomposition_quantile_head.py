# decomposition_quantile_head.py
import math
from typing import List, Dict, Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from modeling_module.models.common_layers.heads.quantile_heads.base_quantile_head import (
    BaseQuantileHead,
    _split_lower_mid_upper,
    _ensure_3d,
)


class _FourierCache:
    """
    Fourier Basis 캐싱 (offset은 호출 시 roll로 처리)
    get(H, K, dtype, device, period) -> [H, 2K]
    """
    def __init__(self):
        # key: (H, K, period, dtype, device)
        self.cache: Dict[Tuple[int, int, int, torch.dtype, torch.device], Tensor] = {}

    def get(self, H: int, K: int, dtype, device, period: int) -> Tensor:
        key = (H, K, period, dtype, device)
        if key in self.cache:
            return self.cache[key]
        t = torch.arange(H, dtype=dtype, device=device)  # [0..H-1]
        feats = []
        P = max(1, int(period))
        for k in range(1, K + 1):
            ang = 2.0 * math.pi * k * (t % P) / P
            feats += [torch.cos(ang), torch.sin(ang)]
        self.cache[key] = torch.stack(feats, dim=-1) if feats else torch.zeros(H, 0, dtype=dtype, device=device)
        return self.cache[key]


class DecompositionQuantileHeadCore(BaseQuantileHead):
    """
    q_mid(t) = PerTime(h_t) + Trend(z; a,b) + Season_t(h_t; θ_t) + Irregular(z)
      - PerTime(h_t): 시간별 중앙값 곡선의 기본형
      - Trend: a + b * (t + offset)
      - Season_t: 시간별 θ_t(h_t) · Fourier(period)   (offset은 기저를 roll)
      - Irregular: 전역 상수
    이후 Δ(softplus 누적)로 하/상 분위수 생성.
    """
    def __init__(
        self,
        in_features: int,
        quantiles: List[float],
        hidden: int = 128,
        dropout: float = 0.0,
        mid: float = 0.5,
        use_trend: bool = True,
        fourier_k: int = 4,
        agg: str = "last",
    ):
        super().__init__(quantiles)
        self.mid = float(mid)
        self.use_trend = bool(use_trend)
        self.fourier_k = max(0, int(fourier_k))
        self.agg = agg
        self.hidden = int(hidden)

        lower, _, upper = _split_lower_mid_upper(self.quantiles, mid=self.mid)
        self.kL, self.kU = len(lower), len(upper)

        if agg not in ("mean", "last"):
            raise ValueError("agg must be 'mean' or 'last'")

        # [B,H,F] → [B,H,hidden]
        self.feat_proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 중앙값: 시간별 회귀
        self.mid_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),  # [B,H,1]
        )

        # Trend/Irregular: 전역 z 기반
        coef_in = hidden
        self.trend_head = nn.Linear(coef_in, 2) if self.use_trend else None   # a, b
        self.irreg_head = nn.Linear(coef_in, 1)

        # Season: 시간별 θ_t(h_t)
        self.season_time_head = (
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2 * self.fourier_k),
            ) if self.fourier_k > 0 else None
        )

        # Δ(softplus 누적)로 하/상 분위수 폭 생성
        outk = self.kL + self.kU
        self.delta_head = nn.Linear(hidden, outk) if outk > 0 else None
        if self.delta_head is not None:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.constant_(self.delta_head.bias, 0.10)
        self.delta_scale = 1.5

        self._fcache = _FourierCache()

    def _pool_feat(self, x_bhf: torch.Tensor):
        # x_bhf: [B, H, F]
        if self.agg == "mean":
            z = x_bhf.mean(dim=1)        # [B, F]
        elif self.agg == "last":
            z = x_bhf[:, -1, :]          # [B, F]
        else:
            raise ValueError("agg must be 'mean' or 'last'")

        h = self.feat_proj(x_bhf)        # [B, H, hidden]
        return z, h

    def forward(self, x: Tensor, *, step_offset: int = 0, period: Optional[int] = None) -> Tensor:
        """
        입력:  x [B,H,F]
        출력: yq [B,3,H]  (Quantile order = [lower... mid ... upper])
        step_offset: IMS에서의 누적 시점 오프셋
        period: 계절 주기(월간=12, 주간=52); None이면 52 기본
        """
        x = _ensure_3d(x)                 # [B,H,F]
        B, H, _ = x.shape
        dtype, device = x.dtype, x.device
        offset = int(step_offset)
        period = int(period) if period is not None else 52

        # 풀링/투영
        z, h = self._pool_feat(x)         # z:[B,F], h:[B,H,hidden]

        # 1) 중앙값: 시간별 h에서 직접 회귀
        q_mid = self.mid_head(h).squeeze(-1)       # [B,H]

        # 2) Trend: a + b * (t + offset)
        if self.use_trend:
            ab = self.trend_head(z)                # [B,2]
            a = ab[:, :1]                          # [B,1]
            b = ab[:, 1:]                          # [B,1]
            t = torch.arange(offset, offset + H, dtype=dtype, device=device).unsqueeze(0)  # [1,H]
            q_mid = q_mid + a + b * t             # [B,H]

        # 3) Season: θ_t(h_t) · Fourier(period)  (기저를 offset만큼 roll)
        if (self.season_time_head is not None) and (self.fourier_k > 0):
            theta_t = self.season_time_head(h)     # [B,H,2K]
            S = self._fcache.get(H, self.fourier_k, dtype, device, period=period)  # [H,2K]
            if offset % period != 0:
                S = S.roll(shifts=offset % period, dims=0)  # 위상 이동
            season = (theta_t * S.unsqueeze(0)).sum(dim=-1)  # [B,H]
            q_mid = q_mid + season

        # 4) Irregular: 전역 상수
        irr = self.irreg_head(z).squeeze(-1)       # [B]
        q_mid = q_mid + irr.unsqueeze(-1)          # [B,H]

        # 5) Δ 누적하여 하/상 분위수 구성
        if self.delta_head is None:
            return torch.stack([q_mid, q_mid, q_mid], dim=1)  # [B,3,H]

        raw = self.delta_head(h)                   # [B,H,kL+kU]
        kL, kU = self.kL, self.kU

        outs = []
        if kL > 0:
            dL = F.softplus(raw[..., :kL]) * self.delta_scale
            dL = torch.cumsum(dL, dim=-1)
            qL = [q_mid - dL[..., i] for i in range(kL - 1, -1, -1)]
            outs.append(torch.stack(qL, dim=1))
        outs.append(q_mid.unsqueeze(1))
        if kU > 0:
            dU = F.softplus(raw[..., kL:]) * self.delta_scale
            dU = torch.cumsum(dU, dim=-1)
            qU = [q_mid + dU[..., i] for i in range(kU)]
            outs.append(torch.stack(qU, dim=1))

        yq = torch.cat(outs, dim=1)                            # [B,3,H]
        return yq


class DecompositionQuantileHead(nn.Module):
    """
    입력: [B,H,F] (또는 [B,1,F])
    출력: [B,3,H]  ← QuantileModel이 기대하는 모양으로 고정
    """
    def __init__(
        self,
        in_features: int,
        quantiles: List[float],
        hidden: int = 256,
        dropout: float = 0.05,
        mid=0.5,
        use_trend: bool = True,
        fourier_k: int = 52,
        agg: str = 'last'
    ):
        super().__init__()
        self.core = DecompositionQuantileHeadCore(
            in_features=in_features,
            quantiles=quantiles,
            hidden=hidden,
            dropout=dropout,
            mid=mid,
            use_trend=use_trend,
            fourier_k=fourier_k,
            agg=agg,
        )

    def forward(self, x: Tensor, *, step_offset: int = 0, period: Optional[int] = None) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Head도 offset/period를 그대로 전달
        yq = self.core(x, step_offset=step_offset, period=period)   # [B,3,H]
        return yq
