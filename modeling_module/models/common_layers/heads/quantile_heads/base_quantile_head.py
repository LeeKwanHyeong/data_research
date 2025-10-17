from typing import List, Tuple

import torch
from torch import nn, Tensor


def _check_and_sort_quantiles(qs: List[float]) -> List[float]:
    """
        분위수 리스트가 0과 1 사이에 있는지 확인하고 정렬된 상태로 반환

        Parameters:
        - qs: 분위수 리스트 (예: [0.5, 0.1, 0.9])

        Returns:
        - 정렬된 분위수 리스트 (예: [0.1, 0.5, 0.9])

        Raises:
        - AssertionError: 분위수가 (0, 1) 구간이 아닐 경우
    """
    assert all(0.0 < q < 1.0 for q in qs), "quantiles must be in (0,1)"
    qs_sorted = sorted(qs)
    return qs_sorted


def _split_lower_mid_upper(qs: List[float], mid: float = 0.5
                           ) -> Tuple[List[float], float, List[float]]:
    """
        분위수 리스트를 중앙값 기준으로 하위/중앙/상위 그룹으로 분할

        Parameters:
        - qs: 분위수 리스트 (예: [0.1, 0.5, 0.9])
        - mid: 중앙값으로 간주할 분위수 (default = 0.5)

        Returns:
        - lower: mid보다 작은 분위수들
        - m: mid 분위수 (없으면 가장 가까운 분위수로 대체됨)
        - upper: mid보다 큰 분위수들
    """
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
    """
        입력 텐서를 [B, H, F] 형태로 강제 변환

        허용 형태:
        - [B, F] → [B, 1, F]
        - [B, H, F] → 그대로 유지

        Parameters:
        - x: 입력 텐서

        Returns:
        - [B, H, F] 형태의 텐서
    """
    # 허용 입력: [B,H,F] 또는 [B,F] (후자는 [B,1,F]로 변환)
    if x.dim() == 2:
        return x.unsqueeze(1)  # [B,1,F]
    assert x.dim() == 3, "Input must be [B,H,F] or [B,F]"
    return x


class BaseQuantileHead(nn.Module):
    """
    Quantile 예측을 위한 베이스 클래스 (추상 클래스)

    공통 인터페이스 정의:
    - 입력: x [B, H, F]
    - 출력: yq [B, Q, H], Q = len(quantiles)

    하위 클래스는 forward() 메서드를 반드시 구현해야 함.
    """
    def __init__(self, quantiles: List[float]):
        """
        Parameters:
        - quantiles: 예측할 분위수 리스트 (예: [0.1, 0.5, 0.9])
        """
        super().__init__()
        self.quantiles = _check_and_sort_quantiles(list(quantiles))

    @torch.no_grad()
    def _check_monotonic(self, yq: Tensor):
        """
        분위수 단조성 검사 (디버깅용)

        분위수 축(Q)을 따라 q10 ≤ q50 ≤ q90 등의 단조성을 만족하는지 확인

        Parameters:
        - yq: 예측 출력 텐서 [B, Q, H]

        Raises:
        - RuntimeError: 분위수 교차(crossing) 발생 시
        """
        assert yq.dim() == 3
        diffs = yq[:, 1:, :] - yq[:, :-1, :]
        if torch.any(diffs < -1e-6):
            raise RuntimeError("Quantile crossing detected.")

    def forward(self, x: Tensor) -> Tensor:
        """
        추상 forward 함수 (서브 클래스에서 구현 필요)

        Parameters:
        - x: 입력 feature [B, H, F]

        Returns:
        - yq: 분위수 예측 결과 [B, Q, H]
        """
        raise NotImplementedError