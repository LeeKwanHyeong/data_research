import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Dict, Tuple

from modeling_module.models.common_layers.heads.quantile_heads.base_quantile_head import BaseQuantileHead, \
    _split_lower_mid_upper, _ensure_3d


class _FourierCache:
    """
    Fourier Basis 캐싱 클래스 (Fourier feature 재사용 최적화용)

    목적:
    - 시점 H에 따라 미리 계산된 sin/cos basis를 dtype/device에 맞춰 캐싱
    - 반복 학습 중 동일 H일 경우 계산 재사용

    get(H, K, dtype, device):
    - H: horizon
    - K: harmonic 개수
    - 반환: [H, 2K] → [cos(2πkt/H), sin(2πkt/H)] 벡터
    """
    def __init__(self):
        self.cache: Dict[Tuple[int, torch.dtype, torch.device], Tensor] = {}

    def get(self, H: int, K: int, dtype, device) -> Tensor:
        """
        Parameters:
        - H: horizon (시계열 길이)
        - K: harmonic 수
        - dtype/device: 현재 입력과 일치하는 타입 및 장치

        Returns:
        - Fourier basis: [H, 2K]
        """
        key = (H, dtype, device)
        if key in self.cache:
            return self.cache[key]
        t = torch.arange(H, dtype=dtype, device=device)  # [0..H-1]
        # [H, 2K]: [cos(2πkt/H), sin(2πkt/H)]
        feats = []
        for k in range(1, K + 1):
            ang = 2.0 * math.pi * k * t / max(1, H)
            feats += [torch.cos(ang), torch.sin(ang)]
        self.cache[key] = torch.stack(feats, dim=-1) if feats else torch.zeros(H, 0, dtype=dtype, device=device)
        return self.cache[key]


class DecompositionQuantileHeadCore(BaseQuantileHead):
    """
    q_mid(t) = Trend(t; θ_T(z)) + Season(t; θ_S(z)) + Irregular(z)
      - Trend: a + b * t  (선택적)
      - Season: Fourier(K)  (H 의존, lazy 캐시)
      - Irregular: z -> scalar via MLP
    이후 Monotone 방식으로 Δ 누적해 임의 분위수 생성.
    """
    def __init__(self, in_features: int, quantiles: List[float],
                 hidden: int = 128, dropout: float = 0.0, mid: float = 0.5,
                 use_trend: bool = True, fourier_k: int = 4,
                 agg: str = "mean"):   # 입력이 [B,H,F]일 때 내부에서 [B,F]로 요약하는 방식
        super().__init__(quantiles)
        self.mid = mid
        self.use_trend = use_trend
        self.fourier_k = int(fourier_k)
        self.agg = agg
        self.quantiles = quantiles
        lower, m, upper = _split_lower_mid_upper(self.quantiles, mid=self.mid)
        self.kL, self.kU = len(lower), len(upper)

        # z 요약 네트워크 ([B,H,F] -> [B,F])
        if agg not in ("mean", "last"):
            raise ValueError("agg must be 'mean' or 'last'")
        self.feat_proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Trend/Season 계수 예측
        coef_in = hidden
        if self.use_trend:
            self.trend_head = nn.Linear(coef_in, 2)   # a, b
        else:
            self.trend_head = None
        self.season_head = nn.Linear(coef_in, 2 * self.fourier_k) if self.fourier_k > 0 else None

        # Irregular (상수 성분)
        self.irreg_head = nn.Linear(coef_in, 1)

        # Δ(누적 softplus)
        outk = self.kL + self.kU
        self.delta_head = nn.Linear(hidden, outk) if outk > 0 else None
        if self.delta_head is not None:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

        self._fcache = _FourierCache()

    def _pool_feat(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: [B,H,F] -> (feat_per_t [B,H,Hid], z [B,Hid])
        """
        h = self.feat_proj(x)  # [B,H,Hid]
        if self.agg == "mean":
            z = h.mean(dim=1)  # [B,Hid]
        else:
            z = h[:, -1, :]    # [B,Hid]
        return h, z

    def forward(self, x: Tensor) -> Tensor:
        x = _ensure_3d(x)                     # [B,H,F]
        B, H, F_ = x.shape
        h, z = self._pool_feat(x)             # [B,H,Hid], [B,Hid]
        dtype, device = x.dtype, x.device

        # q_mid(t)
        q_mid = torch.zeros(B, H, dtype=dtype, device=device)

        if self.use_trend:
            ab = self.trend_head(z)           # [B,2] -> a, b
            a, b = ab[:, :1], ab[:, 1:]
            t = torch.arange(H, dtype=dtype, device=device).unsqueeze(0)  # [1,H]
            q_mid = q_mid + a + b * t

        if self.season_head is not None and self.fourier_k > 0:
            theta = self.season_head(z)       # [B, 2K]
            S = self._fcache.get(H, self.fourier_k, dtype, device)  # [H,2K]
            q_mid = q_mid + torch.matmul(S, theta.T).T  # [B,H]

        # irregular (상수 성분)
        irr = self.irreg_head(z).squeeze(-1)  # [B]
        q_mid = q_mid + irr.unsqueeze(-1)

        # Δ 예측(누적)
        if self.delta_head is None:
            return q_mid.unsqueeze(1)         # [B,1,H]

        raw = self.delta_head(h)              # [B,H,kL+kU]
        kL, kU = self.kL, self.kU

        outs = []
        if kL > 0:
            dL = F.softplus(raw[..., :kL])
            dL = torch.cumsum(dL, dim=-1)
            qL = [q_mid - dL[..., i] for i in range(kL-1, -1, -1)]
            outs.append(torch.stack(qL, dim=1))
        outs.append(q_mid.unsqueeze(1))
        if kU > 0:
            dU = F.softplus(raw[..., kL:])
            dU = torch.cumsum(dU, dim=-1)
            qU = [q_mid + dU[..., i] for i in range(kU)]
            outs.append(torch.stack(qU, dim=1))

        return torch.cat(outs, dim=1)         # [B,Q,H]


class DecompositionQuantileHead(nn.Module):
    """
    기존 decomposition_quantile_head.py 대체:
      - 기존 입력이 [B,D]였다면, 백본단에서 [B,H,F]로 전달되도록 조정하거나,
        아래처럼 [B,D]를 [B,1,D]로 확장하여 사용 가능.
      - 출력: [B,3,H] 필요 시 permute.
    """
    def __init__(self,
                 in_features: int,
                 quantiles: List[float],
                 hidden: int = 128,
                 dropout: float = 0.0,
                 mid = 0.5,
                 use_trend: bool = True,
                 fourier_k: int = 4,
                 agg = 'mean'
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
            agg=agg
        )

    def forward(self, x: Tensor) -> Tensor:
        # [B,D] -> [B,1,D] 허용
        if x.dim() == 2:
            x = x.unsqueeze(1)
        yq = self.core(x)           # [B,3,H]
        return yq.permute(0, 2, 1)  # [B,H,3]