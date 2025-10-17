import math
from typing import List, Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def _check_and_sort_quantiles(qs: List[float]) -> List[float]:
    assert all(0.0 < q < 1.0 for q in qs), "quantiles must be in (0,1)"
    qs_sorted = sorted(qs)
    return qs_sorted


def _split_lower_mid_upper(qs: List[float], mid: float = 0.5
                           ) -> Tuple[List[float], float, List[float]]:
    qs = _check_and_sort_quantiles(qs)
    # mid가 리스트에 없으면 mid에 가장 가까운 분위수를 기준으로 취급
    if mid in qs:
        m = mid
    else:
        m = min(qs, key=lambda x: abs(x - mid))
    lower = [q for q in qs if q < m]
    upper = [q for q in qs if q > m]
    return lower, m, upper


def _ensure_3d(x: Tensor) -> Tensor:
    # 허용 입력: [B,H,F] 또는 [B,F] (후자는 [B,1,F]로 변환)
    if x.dim() == 2:
        return x.unsqueeze(1)  # [B,1,F]
    assert x.dim() == 3, "Input must be [B,H,F] or [B,F]"
    return x


class BaseQuantileHead(nn.Module):
    """
    표준 인터페이스:
      - 입력:  x [B, H, F]
      - 출력:  yq [B, Q, H]  (Q = len(quantiles))
    """
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = _check_and_sort_quantiles(list(quantiles))

    @torch.no_grad()
    def _check_monotonic(self, yq: Tensor):
        # 단조 보장 여부(디버그용)
        # yq: [B, Q, H]  -> Q 축으로 오름차순이어야 함
        assert yq.dim() == 3
        diffs = yq[:, 1:, :] - yq[:, :-1, :]
        if torch.any(diffs < -1e-6):
            raise RuntimeError("Quantile crossing detected.")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError