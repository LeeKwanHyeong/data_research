import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Tuple

from modeling_module.models.common_layers.heads.quantile_heads.base_quantile_head import BaseQuantileHead, \
    _split_lower_mid_upper, _ensure_3d


class MonotoneQuantileHeadCore(BaseQuantileHead):
    """
    - 입력:  [B, H, F]
    - 출력:  [B, Q, H]
    - 전략: q_mid(≈0.5)를 직접 예측, 좌/우 Δ를 softplus로 양수화 후 누적합(cumsum)하여 단조 보장
    """
    def __init__(self, in_features: int, quantiles: List[float],
                 hidden: int = 128, dropout: float = 0.0, mid: float = 0.5):
        super().__init__(quantiles)
        self.mid = mid
        self.quantiles = quantiles
        lower, m, upper = _split_lower_mid_upper(self.quantiles, mid=self.mid)
        self.lower = lower
        self.mid_eff = m
        self.upper = upper

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, 1)  # q_mid

        # 누적 Δ 개수
        self.k_lower = len(self.lower)
        self.k_upper = len(self.upper)
        outk = self.k_lower + self.k_upper
        self.delta_head = nn.Linear(hidden, outk) if outk > 0 else None

        # 초기에는 폭이 0에 가깝도록(안정)
        if self.delta_head is not None:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = _ensure_3d(x)              # [B,H,F]
        B, H, F_ = x.shape
        h = self.net(x)                # [B,H,Hid]
        mu = self.mu(h).squeeze(-1)    # [B,H]  (q_mid)

        if self.delta_head is None:
            # 분위수가 mid 하나뿐인 경우
            yq = mu[:, None, :]        # [B,1,H]
            return yq

        raw = self.delta_head(h)       # [B,H,(kL+kU)]
        if self.k_lower > 0:
            dL = F.softplus(raw[..., :self.k_lower])         # [B,H,kL]
            dL = torch.cumsum(dL, dim=-1)                    # 누적
        else:
            dL = None
        if self.k_upper > 0:
            dU = F.softplus(raw[..., self.k_lower:])         # [B,H,kU]
            dU = torch.cumsum(dU, dim=-1)
        else:
            dU = None

        # [B,H] 기준으로 좌/우 분위수 생성
        outs = []
        if dL is not None:
            # 낮은 분위수는 mu - 누적Δ (역순 매핑 주의)
            qL = torch.stack([mu - dL[..., i] for i in range(self.k_lower - 1, -1, -1)], dim=1)  # [B,kL,H]
            outs.append(qL)
        # 중앙
        outs.append(mu.unsqueeze(1))  # [B,1,H]
        if dU is not None:
            qU = torch.stack([mu + dU[..., i] for i in range(self.k_upper)], dim=1)              # [B,kU,H]
            outs.append(qU)

        yq = torch.cat(outs, dim=1)   # [B,Q,H]
        return yq

class MonotoneQuantileHead(nn.Module):
    """
    기존 monotone_quantile_head.py 대체:
      - 입력:  [B,H,F] (기존 [B,H,F]과 동일)
      - 출력:  [B,H,3] 대신 [B,3,H] (필요 시 .permute(0,2,1) 적용)
    """
    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.core = MonotoneQuantileHeadCore(
            in_features=in_features,
            quantiles=[0.1, 0.5, 0.9],
            hidden=hidden,
            dropout=dropout,
            mid=0.5
        )

    def forward(self, x: Tensor) -> Tensor:
        yq = self.core(x)           # [B,3,H]
        return yq.permute(0, 2, 1)  # [B,H,3]  (기존 출력과 맞추려면)