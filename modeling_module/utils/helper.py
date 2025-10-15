import torch
from typing import Tuple, Dict

'''
coverage_per_q는 q와 가까워야 좋음.
예: q=0.1 → 0.10±0.02 정도면 양호, 0.03 이하면 언더/오버 심각.
'''
@torch.no_grad()
def quantile_coverage(
    pred_q: torch.Tensor,  # (B,Q,H)
    y: torch.Tensor,       # (B,H)
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    reduce: str = "overall"  # "overall" | "per_h" | "per_q"
) -> Dict[str, torch.Tensor]:
    """
    각 분위수 q에 대해 coverage = mean( y <= y_hat_q ).
    reduce:
      - "overall": 스칼라(분위수별)
      - "per_h"  : (H,) 각 시점별 coverage
      - "per_q"  : (Q,) 각 분위수별 coverage (overall)
    """
    B, Q, H = pred_q.shape
    y_exp = y.unsqueeze(1).expand(B, Q, H)  # (B,Q,H)
    hit = (y_exp <= pred_q).float()         # (B,Q,H)

    if reduce == "per_h":
      cov_h = hit.mean(dim=0).mean(dim=0)   # (H,)
      return {"coverage_per_h": cov_h}      # 시점별 평균(분위수/배치 평균)

    # 분위수별 overall
    cov_q = hit.mean(dim=(0, 2))  # (Q,)  (B,H 평균)
    out = {"coverage_per_q": cov_q}
    if reduce == "overall":
      # 스칼라 dict (분위수별 스칼라라서 텐서(Q,) 유지)
      return out
    elif reduce == "per_q":
      return out
    else:
      return out

'''
interval80_cov는 0.8 근처, interval80_wid는 도메인에 비춰 작을수록 좋지만 coverage가 떨어지면 의미 없음.
per_h를 보면 초반/중반/후반에서 어디가 무너지는지 파악 가능. 후반 언더가 크면 horizon-decay/regularization을 조절.
'''
@torch.no_grad()
def interval_coverage_width(
    pred_q: torch.Tensor,  # (B,Q,H)
    y: torch.Tensor,       # (B,H)
    lower_q: float = 0.1,
    upper_q: float = 0.9,
    reduce: str = "overall"  # "overall" | "per_h"
) -> Dict[str, torch.Tensor]:
    """
    중앙 분위수 대칭이든 아니든, 지정한 (lower_q, upper_q) 구간에 대해:
      - coverage: mean( y in [qL, qU] )
      - width   : mean( qU - qL )
    """
    Qs = pred_q.size(1)
    # 인덱스 찾기 (quantile 리스트를 따로 주면 그에 맞춰 인덱싱하세요)
    # 여기선 간단히 정렬된 분위수라고 가정하고 가장 가까운 쪽을 사용
    def _closest_idx(q):
        grid = torch.linspace(0, 1, Qs, device=pred_q.device)
        return int(torch.argmin((grid - q).abs()).item())
    iL = _closest_idx(lower_q)
    iU = _closest_idx(upper_q)

    qL = pred_q[:, iL, :]  # (B,H)
    qU = pred_q[:, iU, :]  # (B,H)
    y_in = ((y >= qL) & (y <= qU)).float()   # (B,H)
    width = (qU - qL)                        # (B,H)

    if reduce == "per_h":
        cov_h = y_in.mean(dim=0)             # (H,)
        wid_h = width.mean(dim=0)            # (H,)
        return {"interval_cov_per_h": cov_h, "interval_wid_per_h": wid_h}
    else:
        cov = y_in.mean()                    # scalar
        wid = width.mean()                   # scalar
        return {"interval_cov": cov, "interval_wid": wid}
