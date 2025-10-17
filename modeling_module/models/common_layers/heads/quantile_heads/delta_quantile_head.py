import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List

from modeling_module.models.common_layers.heads.quantile_heads.base_quantile_head import _split_lower_mid_upper, \
    _ensure_3d, _check_and_sort_quantiles


class DeltaQuantileHeadCore(nn.Module):
    """
    - 역할: q_mid 주변의 좌/우 Δ만 예측(softplus + 누적) → 교차 방지
    - 입력: [B,H,F]
    - 출력: (dL, dU)
        dL: [B,H,kL]  (양수 누적Δ; 낮은 분위수는 q_mid - cumsum(dL)[::-1])
        dU: [B,H,kU]  (양수 누적Δ; 높은 분위수는 q_mid + cumsum(dU))
    """
    def __init__(self, in_features: int, quantiles: List[float],
                 hidden: int = 128, dropout: float = 0.0, mid: float = 0.5):
        super().__init__()
        lower, m, upper = _split_lower_mid_upper(quantiles, mid=mid)
        self.kL, self.kU = len(lower), len(upper)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        outk = self.kL + self.kU
        self.delta_head = nn.Linear(hidden, outk) if outk > 0 else None
        if self.delta_head is not None:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

    def forward(self, x: Tensor) -> tuple[Tensor | None, Tensor | None]:
        x = _ensure_3d(x)
        h = self.net(x)
        if self.delta_head is None:
            return None, None
        raw = self.delta_head(h)
        dL = F.softplus(raw[..., :self.kL]) if self.kL > 0 else None
        dU = F.softplus(raw[..., self.kL:]) if self.kU > 0 else None
        # 누적(단조)
        if dL is not None: dL = torch.cumsum(dL, dim=-1)
        if dU is not None: dU = torch.cumsum(dU, dim=-1)
        return dL, dU


def compose_quantiles(q_mid: Tensor,
                      dL: Tensor | None,
                      dU: Tensor | None,
                      quantiles: List[float],
                      mid: float = 0.5) -> Tensor:
    """
    q_mid: [B,H] 또는 [B,H,1]
    dL:    [B,H,kL] (누적Δ, 양수)  -> 낮은 분위수는 q_mid - dL[::-1]
    dU:    [B,H,kU] (누적Δ, 양수)  -> 높은 분위수는 q_mid + dU
    반환:  [B,Q,H]
    """
    quantiles = _check_and_sort_quantiles(quantiles)
    lower, m, upper = _split_lower_mid_upper(quantiles, mid=mid)

    if q_mid.dim() == 3:
        q_mid = q_mid.squeeze(-1)
    # 낮은/높은 분위수 생성
    outs = []
    if len(lower) > 0:
        assert dL is not None, "lower deltas required"
        kL = dL.shape[-1]
        qL = [q_mid - dL[..., i] for i in range(kL-1, -1, -1)]
        outs.append(torch.stack(qL, dim=1))     # [B,kL,H]
    outs.append(q_mid.unsqueeze(1))             # [B,1,H]
    if len(upper) > 0:
        assert dU is not None, "upper deltas required"
        kU = dU.shape[-1]
        qU = [q_mid + dU[..., i] for i in range(kU)]
        outs.append(torch.stack(qU, dim=1))     # [B,kU,H]
    return torch.cat(outs, dim=1)               # [B,Q,H]


class DeltaQuantileHead(nn.Module):
    """
    기존 delta_quantile_head.py 대체:
      - 입력: [B,H,F]
      - 출력: Δ_minus/Δ_plus 형태 필요 시 아래 방식으로 조합
    """
    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.core = DeltaQuantileHeadCore(
            in_features=in_features,
            quantiles=[0.1, 0.5, 0.9],
            hidden=hidden,
            dropout=dropout,
            mid=0.5
        )

    def forward(self, x: Tensor):
        dL, dU = self.core(x)
        return dL, dU