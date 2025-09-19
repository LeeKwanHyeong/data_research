from typing import Optional, Tuple
import torch.nn.functional as F

import torch

__all__ = [
    "pinball_plain",
    "pinball_loss_weighted",
    "pinball_loss_weighted_masked",
    "intermittent_weights",
    "intermittent_weights_balanced",
    "intermittent_pinball_loss",     # 멀티-분위수용 간헐 가중 핀볼 래퍼
    "intermittent_point_loss",       # 점추정(MAE/MSE/Huber/단일핀볼)용 간헐 가중 래퍼
    "intermittent_nll_loss",         # 분포형(NLL)용 간헐 가중 래퍼
    "horizon_decay_weights",     # 호라이즌 감쇠 가중
    "newsvendor_q_star",         # 비용기반 q* 계산
]

# -------------------------------------------------------
# zero-run 길이 계산: 샘플링 과정 중, 0 수요들에 대한 길이 계산
# -------------------------------------------------------
def _zero_runlength(y: torch.Tensor) -> torch.Tensor:
    """
    y: (B, H) (>=0)
    return: (B, H) 현재 시점까지의 연속 0 길이 (포함)
    """

    B, H = y.shape
    r = torch.zeros_like(y)
    for t in range(H):
        if t == 0:
            r[:, t] = (y[:, t] == 0).to(y.dtype)
        else:
            r[:, t] = torch.where(y[:, t] == 0, r[:, t - 1] + 1.0, torch.zeros_like(y[:, t]))
    return r

# -------------------------------------------------------
# 간헐 수요 가중 손실 기본: Intermittent Demand Weighted Loss
# -------------------------------------------------------
def intermittent_weights(
        target: torch.Tensor,
        alpha_zero: float = 1.2,
        alpha_pos: float = 1.0,
        gamma_run: float = 0.6,
        cap: Optional[float] = None,
) -> torch.Tensor:
    """
    w = α_zero·I[y=0]·(1 + γ·runlen_norm) + α_pos·I[y>0]
    cap: (Optional) 너무 큰 가중치 폭주 방지 상한
    """
    run = _zero_runlength(target)
    norm_run = run / max(1, target.size(1))
    is_zero = (target == 0).to(target.dtype)
    is_pos = 1.0 - is_zero
    w = alpha_zero * is_zero * (1.0 + gamma_run * norm_run) + alpha_pos * is_pos
    if cap is not None:
        w = torch.clamp(w, max = float(cap))
    return w

def intermittent_weights_balanced(target: torch.Tensor,
                                   alpha_zero: float = 1.2,
                                   alpha_pos: float = 1.0,
                                   gamma_run: float = 0.6,
                                   clip_run: float = 0.5,
                                   eps: float = 1e-8) -> torch.Tensor:
    """
    zero/positive 각각 평균 1로 정규화된 가중.
    연속 0 보너스는 norm 후 clip_run 상한으로 제한해 과도한 0-편향 예방.
    """
    # run-length
    B, H = target.shape
    run = torch.zeros_like(target)
    for t in range(H):
        if t == 0:
            run[:, t] = (target[:, t] == 0).to(target.dtype)
        else:
            run[:, t] = torch.where(target[:, t] == 0, run[:, t-1] + 1.0, torch.zeros_like(run[:, t-1]))

    norm_run = (run / max(1, H)).clamp(max=clip_run)

    is_zero = (target == 0).to(target.dtype)
    is_pos  = 1.0 - is_zero

    w_zero = alpha_zero * is_zero * (1.0 + gamma_run * norm_run)
    w_pos  = alpha_pos  * is_pos

    m_zero = (w_zero.sum() / (is_zero.sum() + eps)).clamp_min(eps)
    m_pos  = (w_pos.sum()  / (is_pos.sum()  + eps)).clamp_min(eps)

    w = (w_zero / m_zero) + (w_pos / m_pos)
    return w

def pinball_plain(pred_q, y, quantiles=(0.1, 0.5, 0.9)):
    diff = y.unsqueeze(1) - pred_q  # (B,Q,H)
    losses = [torch.maximum(q*diff[:,i], (q-1)*diff[:,i]).mean()
              for i, q in enumerate(quantiles)]
    return sum(losses) / len(losses)


def pinball_loss_weighted_masked(pred_q: torch.Tensor,
                                  target: torch.Tensor,
                                  quantiles: Tuple[float, ...],
                                  weights: torch.Tensor | None = None,
                                  eps: float = 1e-8) -> torch.Tensor:
    """
    하위 분위수(q<0.5)에만 가중 적용. 나머지는 비가중 평균.
    pred_q: (B,Q,H), target: (B,H), weights: (B,H)
    """
    diff = target.unsqueeze(1) - pred_q
    losses = []
    for i, q in enumerate(quantiles):
        e = torch.maximum(q * diff[:, i], (q - 1.0) * diff[:, i])  # (B,H)
        if (weights is not None) and (q < 0.5):
            e = (e * weights).sum() / (weights.sum() + eps)
        else:
            e = e.mean()
        losses.append(e)
    return sum(losses) / len(losses)

# -------------------------------------------------------
# 멀티-분위수 예측용: Weighted Pinball Loss
# -------------------------------------------------------
def pinball_loss_weighted(
        pred_q: torch.Tensor,
        target: torch.Tensor,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        weights: torch.Tensor | None = None,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    pred_q: (B, Q, H)
    target: (B, H)
    weights: (B, H) or None
    """
    B, Q, H = pred_q.shape
    diff = target.unsqueeze(1) - pred_q # (B, Q, H)
    losses = []
    for i, q in enumerate(quantiles):
        e = torch.maximum(q * diff[:, i], (q - 1) * diff[:, i]) # (B, H)
        if weights is not None:
            e = e * weights
            denom = torch.clamp(weights.mean(), min = eps)
            losses.append(e.mean() / denom)
        else:
            losses.append(e.mean())
    return sum(losses) / len(losses)

# -------------------------------------------
# 호라이즌 감쇠 가중: 가까운 시점 가중↑
# -------------------------------------------
def horizon_decay_weights(target_like: torch.Tensor, tau_h: float = 24.0) -> torch.Tensor:
    """
    target_like: (B, H) or (H,)의 dtype/device/shape 기준
    tau_h: 클수록 완만, 작을수록 급격 감쇠
    return: (B, H) (입력이 (H,)면 broadcasting 가능)
    """
    if target_like.dim() == 1:
        H = target_like.size(0)
        device, dtype = target_like.device, target_like.dtype
        h_idx = torch.arange(H, device = device, dtype = dtype)
        w = torch.exp(-h_idx / float(tau_h)) # (H,)
        return w

    elif target_like.dim() == 2:
        B, H = target_like.shape
        device, dtype = target_like.device, target_like.dtype
        h_idx = torch.arange(H, device = device, dtype = dtype)
        w = torch.exp(-h_idx / float(tau_h)).unsqueeze(0).expand(B, H) # (B, H)
        return w
    else:
        raise ValueError('target_like must be (H,) or (B, H)')

# -------------------------------------------
# 비용 기반 q* (Newsvendor): q* = Cu / (Cu + Co)
# -------------------------------------------
def newsvendor_q_star(Cu: float, Co: float, eps: float = 1e-8) -> float:
    return float(Cu) / max(float(Cu) + float(Co), eps)

# -------------------------------------------------------
# 멀티-분위수 예측용: Intermittent Weighted Pinball Loss
# -------------------------------------------------------
def intermittent_pinball_loss(
        pred_q: torch.Tensor,
        target: torch.Tensor,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        alpha_zero: float = 1.2,
        alpha_pos: float = 1.0,
        gamma_run: float = 0.6,
        cap: Optional[float] = None,
        use_horizon_decay: bool = False,
        tau_h: float = 24.0,
        eps: float = 1e-8,
) -> torch.Tensor:
    '''
    intermittent weighted + horizon decay 를 결합한 pinball loss wrapper.
    '''
    w = intermittent_weights(
        target,
        alpha_zero = alpha_zero,
        alpha_pos = alpha_pos,
        gamma_run = gamma_run,
        cap = cap
    ) # (B, H)

    if use_horizon_decay:
        w = w * horizon_decay_weights(target, tau_h = tau_h)

    return pinball_loss_weighted(pred_q, target, quantiles = quantiles, weights = w, eps = eps)

# -------------------------------------------------------
# 점추정 예측용: Intermittent Weighted Point Loss
# Ex) Titan/PatchTST
#    (MAE / MSE / Huber / 단일 핀볼(q=tau))
# -------------------------------------------------------
def intermittent_point_loss(
        y_hat: torch.Tensor,
        target: torch.Tensor,
        mode: str = 'mae',
        tau: float = 0.5,
        delta: float = 1.0,
        alpha_zero: float = 1.2,
        alpha_pos: float = 1.0,
        gamma_run: float = 0.5,
        cap: Optional[float] = None,
        use_horizon_decay: bool = False,
        tau_h: float = 24.0,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    점추정 모델(Titan/PatchTST 기본 헤드 등) 에 간헐 가중을 적용한 손실.
    - mode = 'pinball'이면 q=tau를 최적화 (비용기반 q* 사용 가능).
    """
    w = intermittent_weights(
        target,
        alpha_zero = alpha_zero,
        alpha_pos = alpha_pos,
        gamma_run = gamma_run,
        cap = cap
    ) # (B, H)
    if use_horizon_decay:
        w = w * horizon_decay_weights(target, tau_h = tau_h)

    if mode == 'mae':
        e = (y_hat - target).abs()  # (B, H)
    elif mode == 'mse':
        e = (y_hat - target).pow(2) # (B, H)
    elif mode == 'huber':
        # reduction='none'로 시점별 손실 유지
        e = F.huber_loss(y_hat, target, delta = delta, reduction = 'none') # (B, H)
    elif mode == 'pinball':
        diff = target - y_hat
        e = torch.maximum(tau * diff, (tau - 1.0) * diff) # (B, H)
    else:
        raise ValueError("mode must be one of ['mae', 'mse', 'huber', 'pinball']")

    # weighted mean
    denom = torch.clamp(w.mean(), min = eps)
    return (e * w).mean() / denom

# -------------------------------------------------------
# 분포형(NLL) 모델용: Intermittent Weighted NLL
# -------------------------------------------------------
def intermittent_nll_loss(
        nll: torch.Tensor,  # (B, H) = -log p(y | theta)
        target: torch.Tensor,
        alpha_zero: float = 1.2,
        alpha_pos: float = 1.0,
        gamma_run: float = 0.6,
        cap: Optional[float] = None,
        use_horizon_decay: bool = False,
        tau_h: float = 24.0,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    DeepAR/Tweedie/ZI-NB 등 분포형 모델의 NLL에 간헐 가중 적용.
    """
    w = intermittent_weights(
        target,
        alpha_zero = alpha_zero,
        alpha_pos = alpha_pos,
        gamma_run = gamma_run,
        cap = cap
    )
    if use_horizon_decay:
        w = w * horizon_decay_weights(target, tau_h = tau_h)

    denom = torch.clamp(w.mean(), min = eps)
    return (nll * w).mean() / denom

