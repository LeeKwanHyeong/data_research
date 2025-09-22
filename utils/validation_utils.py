import torch
from typing import Tuple, Dict
from utils.custom_loss_utils import pinball_loss_weighted

# Data Loader validation check
def collect_indices(loader, max_batches=999999):
    idxs = []
    seen=0
    for b in loader:
        # 데이터셋에서 원본 인덱스를 함께 반환하도록 구현되어 있지 않다면,
        # 배치 단위로 해시를 만들어 임시로 비교합니다 (완벽하진 않음)
        x = b[0]
        idxs.append(torch.tensor([x.numel(), x.sum()]).float())  # 조잡한 지문
        seen += 1
        if seen>=max_batches: break
    return torch.stack(idxs)

def _smape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    sMAPE = mean( 2*|pred - target| / (|pred| + |target| + eps))
    pred, target: (B, H)
    """
    return (2.0 * (pred - target).abs() / (pred.abs() + target.abs() + eps)).mean()

@torch.no_grad()
def compute_validation_metrics_for_patchmixer(
        model: torch.nn.Module,
        val_loader,
        device: str | torch.device = 'cpu',
        *,
        q_use: float = 0.5,
) -> Dict[str, float]:
    from model_runner.inferences.patch_mixer_inference import evaluate_with_truth

    """
    PatchMixer (분위수 헤드) 모델의 밸리데이션 지표를 한 번에 계산.

    계산 항목
    --------
    - MAE (P50 기준 점추정)
    - RMSE (P50 기준)
    - sMAPE (P50 기준)
    - Pinball(avg): (Q=0.1,0.5,0.9) 평균 핀볼 로스
    - COV@q: 경험적 커버리지 (y_true <= q-예측) @ q∈{0.1, 0.5, 0.9}
      * 로그에서 보던 COV[q]와 동일 개념
    - Coverage@80: P10~P90 구간 커버리지 (빈도)
    - Width@80: P90 - P10 평균 폭

    Notes
    -----
    - 내부에서 evaluate_PatchMixer_with_truth(..., return_full=True)로 (B,Q,H) 예측을 얻습니다.
    - 모델이 P10,P50,P90 순서로 예측한다고 가정합니다. (head.qs = (0.1,0.5,0.9))
    - 반환은 dict 형태로, 바로 로그/프린트에 사용하기 좋습니다.

    Parameters
    ----------
    model : torch.nn.Module
        PatchMixerQuantileModel (DecompQuantileHead 등) 인스턴스
    val_loader : DataLoader
        검증용 로더. batch는 (x, y, part) 형태를 가정
    device : str | torch.device
        연산 디바이스
    q_use : float
        필요시 중앙 점추정 외 다른 분위수 점수를 추가로 보고 싶을 때 사용 가능 (현재는 P50 지표)

    Returns
    -------
    Dict[str, float]
        {
          "MAE": ...,
          "RMSE": ...,
          "sMAPE": ...,
          "Pinball(avg)": ...,
          "COV@0.1": ...,
          "COV@0.5": ...,
          "COV@0.9": ...,
          "Coverage@80": ...,
          "Width@80": ...
        }
    """
    # 1) (B, Q, H) 예측과 (B, H, 1) GT 가져오기
    preds_q, trues, _ = evaluate_with_truth(
        model, val_loader, device = device, q_use = q_use, return_full = True
    )   # preds_q: (B, Q, H), trues: (B, H, 1)

    # 2) Tensor 정리
    preds_q = preds_q.to(torch.float32)
    y_true = trues.squeeze(-1).to(torch.float32)    # (B, H)
    assert preds_q.dim() == 3, f"Expected (B, Q, H), got shape = {tuple(preds_q.size())}"
    B, Q, H = preds_q.shape
    assert Q >= 3, "Expected at least 3 quantiles (0.1, 0.5, 0.9)."

    # 0: P10, 1: P50, 2: P90
    q10 = preds_q[:, 0, :]
    q50 = preds_q[:, 1, :]
    q90 = preds_q[:, 2, :]

    # 3) Point Estimator Validation Metrics (P50)
    mae = (q50 - y_true).abs().mean().item()
    rmse = torch.sqrt(((q50 - y_true) ** 2).mean()).item()
    smape = _smape(q50, y_true).item()

    # 4) Statistics Metrics
    # 4-1) pinball (Q = 0.1, 0.5, 0.9)
    pinball = pinball_loss_weighted(
        preds_q, y_true, quantiles=(0.1, 0.5, 0.9), weights=None
    ).item()

    # 4-2) empirical coverage each q (Cov[q] = mean(y_true <= q_pred))
    cov_q10 = (y_true <= q10).float().mean().item()
    cov_q50 = (y_true <= q50).float().mean().item()
    cov_q90 = (y_true <= q90).float().mean().item()

    # 4-3) 80% forecast interval coverage
    coverage_80 = ((y_true >= q10) & (y_true <= q90)).float().mean().item()
    avg_interval_width = (q90 - q10).mean().item()

    return {
        'MAE': mae,
        'RMSE': rmse,
        'sMAPE': smape,
        'Pinball(avg)': pinball,
        'Coverage@0.1': cov_q10,
        'Coverage@0.5': cov_q50,
        'Coverage@0.9': cov_q90,
        'Coverage@80': coverage_80,
        'Width@80': avg_interval_width,
    }