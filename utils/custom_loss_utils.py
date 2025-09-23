from typing import Optional, Tuple, Iterable
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
    "horizon_decay_weights",         # 호라이즌 감쇠 가중
    "newsvendor_q_star",             # 비용기반 q* 계산
]

# =====================================================
# Core helpers (공통 유틸)
# =====================================================
def _ensure_float(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_floating_point() else x.float()

def _weighted_mean(e: torch.Tensor, w: Optional[torch.Tensor], eps: float = 0.8) -> torch.Tensor:
    """
    e: (...), w: same shape or broadcastable to e
    """
    if w is None:
        return e.mean()
    e = e * w
    denom = torch.clamp(w.mean(), min = eps)
    return e.mean() / denom

def _apply_horizon_decay(w: torch.Tensor, target_like: torch.Tensor, tau_h: float = 24.0) -> torch.Tensor:
    """
    W, target_like: (B, H) 동일 shape 가정 (or broadcastable)
    """
    B, H = target_like.shape
    device, dtype = target_like.device, target_like.dtype
    h_idx = torch.arange(H, device = device, dtype = dtype)
    decay = torch.exp(-h_idx / float(tau_h)).unsqueeze(0) # (1, H)
    return w * decay

def _pinball_elem(diff: torch.Tensor, q: float) -> torch.Tensor:
    # element-wise pinball for a given quantile q
    # diff = target - pred
    return torch.maximum(q * diff, (q - 1.0) * diff)

def _zero_runlength(target: torch.Tensor) -> torch.Tensor:
    """
    target: (B, H) (>= 0 가정)
    return: (B, H) 현재 시점까지의 연속 0 길이 (포함)
    """
    target = _ensure_float(target)
    B, H = target.shape
    r = torch.zeros_like(target)
    for t in range(H):
        if t == 0:
            r[:, t] = (target[:, t] == 0).to(target.dtype)
        else:
            r[:, t] = torch.where(target[:, t] == 0, r[:, t-1] + 1.0, torch.zeros_like(target[:, t]))
    return r

def _intermittent_weights_core(
        target: torch.Tensor,
        alpha_zero: float,
        alpha_pos: float,
        gamma_run: float,
        cap: Optional[float] = None,
        *,
        balanced: bool = False,
        clip_run: float = 0.5,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Common Base: sparse(0 - run) based weight.
    - balanced = False: default
    - balanced = True : Normalize zero/positive to 1 each
    """
    target = _ensure_float(target)
    B, H = target.shape
    run = _zero_runlength(target)
    norm_run = (run / max(1, H))
    if balanced:
        norm_run = norm_run.clamp(max = clip_run) # Max 0.5 Min: -inf

    is_zero = (target == 0).to(target.dtype)
    is_pos = 1.0 - is_zero

    w_zero = alpha_zero * is_zero * (1.0 + gamma_run * norm_run)
    w_pos = alpha_pos * is_pos

    if balanced:
        # Normalize zero/pos to 1 each
        m_zero = (w_zero.sum() / (is_zero.sum() + eps)).clamp_min(eps)
        m_pos = (w_pos.sum() / (is_pos.sum() + eps)).clamp_min(eps)
        w = (w_zero / m_zero) + (w_pos / m_pos)
    else:
        w = w_zero + w_pos
        if cap is not None:
            w = torch.clamp(w, max = float(cap))

    return w

# =====================================================
# Intermittent weights
# =====================================================
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
    return _intermittent_weights_core(
        target, alpha_zero, alpha_pos, gamma_run, cap, balanced = False
    )

def intermittent_weights_balanced(
        target: torch.Tensor,
        alpha_zero: float = 1.2,
        alpha_pos: float = 1.0,
        gamma_run: float = 0.6,
        clip_run: float = 0.5,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    zero/Normalized weighted with a positive mean of 1.
    continuous zeros are bounded by the upper bound of Clip_run after Norm.
    """
    return _intermittent_weights_core(
        target, alpha_zero, alpha_pos, gamma_run,
        cap = None, balanced = True, clip_run = clip_run, eps = eps
    )

# =====================================================
# Pinball losses
# =====================================================
def pinball_plain(pred_q: torch.Tensor,
                  y: torch.Tensor,
                  quantiles: Iterable[float] = (0.1, 0.5, 0.9)) -> torch.Tensor:
    """
    pred_q: (B, Q, H), y: (B, H)
    Pinball mean of all quantiles.
    """
    y = _ensure_float(y)
    pred_q = _ensure_float(pred_q)
    diff = y.unsqueeze(1) - pred_q # (B, Q, H)
    losses = []
    for i, q in enumerate(quantiles):
        losses.append(_pinball_elem(diff[:, i], float(q)).mean())
    return sum(losses) / max(1, len(losses))

def pinball_loss_weighted(
        pred_q: torch.Tensor,
        target: torch.Tensor,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    pred_q: (B, Q, H)
    target: (B, H)
    weights: (B, H) or broadcastable to (B, 1, H)
    """
    pred_q = _ensure_float(pred_q)
    target = _ensure_float(target)

    diff = target.unsqueeze(1) - pred_q # (B, Q, H)
    losses = []
    for i, q in enumerate(quantiles):
        e = _pinball_elem(diff[:, i], float(q))
        if weights is None:
            losses.append(e.mean())

        else:
            # weights: (B, H) -> broadcast to (B, H)
            losses.append(_weighted_mean(e, weights, eps = eps))
    return sum(losses) / max(1, len(quantiles))

def pinball_loss_weighted_masked(
        pred_q: torch.Tensor,
        target: torch.Tensor,
        quantiles: Tuple[float, ...],
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    하위 분위수 (q<0.5)에만 가중 적용. 나머지는 비가중 평균.
    pred_q: (B, Q, H), target: (B, H), weights: (B, H) or None
    """
    pred_q = _ensure_float(pred_q)
    target = _ensure_float(target)

    diff = target.unsqueeze(1) - pred_q
    losses = []
    for i, q in enumerate(quantiles):
        e = _pinball_elem(diff[:, i], float(q))
        # 하위 q < 0.5
        if (weights is not None) and (q < 0.5):
            losses.append(_weighted_mean(e, weights, eps))
        else:
            losses.append(e.mean())
    return sum(losses) / max(1, len(quantiles))

# =====================================================
# Horizon decay weights
# =====================================================
def horizon_decay_weights(target_like: torch.Tensor, tau_h: float = 24.0) -> torch.Tensor:
    """
    target_like: (B, H) or (H,) 의 dtype/device/shape 기준
    tau_h: 클수록 완만, 작을수록 급격 감쇠
    return: (B, H) or (H,) 입력과 동일한 rank로 변환
    """
    if target_like.dim() == 1:
        H = target_like.size(0)
        device, dtype = target_like.device, target_like.dtype
        h_idx = torch.arange(H, device = device, dtype = dtype)
        return torch.exp(-h_idx / float(tau_h)) # (H,)
    elif target_like.dim() == 2:
        B, H = target_like.shape
        device, dtype = target_like.device, target_like.dtype
        h_idx = torch.arange(H, device = device, dtype = dtype)
        return torch.exp(-h_idx / float(tau_h)).unsqueeze(0).expand(B, H) # (B, H)
    else:
        raise ValueError('target_like must be (H,) or (B,H)')


# =====================================================
# Newsvendor q*
# =====================================================
def newsvendor_q_star(Cu: float, Co: float, eps: float = 1e-8) -> float:
    return float(Cu) / max(float(Cu) + float(Co), eps)

# =====================================================
# Intermittent wrappers
# =====================================================
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
    """
    intermittent weight
    + (optional) multi quantile pinball wrapper combined with horizon decay
    """
    w = intermittent_weights(
        target, alpha_zero = alpha_zero, alpha_pos = alpha_pos, gamma_run = gamma_run, cap = cap
    ) # (B, H)
    if use_horizon_decay:
        w = _apply_horizon_decay(w, target, tau_h = tau_h)
    return pinball_loss_weighted(pred_q, target, quantiles = quantiles, weights = w, eps = eps)

def intermittent_point_loss(
        y_hat: torch.Tensor,
        target: torch.Tensor,
        mode: str = 'mae',      # 'mae' | 'mse' | 'huber' | 'pinball'
        tau: float = 0.5,       # pinball q
        delta: float = 1.0,     # huber delta
        alpha_zero: float = 1.2,
        alpha_pos: float = 1.0,
        gamma_run: float = 0.5,
        cap: Optional[float] = None,
        use_horizon_decay: bool = False,
        tau_h: float = 24.0,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    점추정 모델(Titan/self_supervised 등)에 간헐 가중을 적용한 손실.
    - mode='pinball'이면 τ-quantile 최적화(서비스 비용 기반 q* 사용 가능).
    """
    y_hat = _ensure_float(y_hat)
    target = _ensure_float(target)

    w = intermittent_weights(
        target, alpha_zero = alpha_zero, alpha_pos = alpha_pos, gamma_run = gamma_run,
        cap = cap
    ) # (B, H)
    if use_horizon_decay:
        w = _apply_horizon_decay(w, target, tau_h = tau_h)

    if mode == 'mae':
        e = (y_hat - target).abs()
    elif mode == 'mse':
        e = (y_hat - target).pow(2)
    elif mode == 'huber':
        e = F.huber_loss(y_hat, target, delta = delta, reduction = 'none')
    elif mode == 'pinball':
        diff = target - y_hat
        e = _pinball_elem(diff, float(tau))
    else:
        raise ValueError("mode must be ['mae', 'mse', 'huber', 'pinball']")

def intermittent_nll_loss(
        nll: torch.Tensor,
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
    DeepAR/Tweedie/ZI-NB 등 분포형 모델의 NLL에 간헐 가중 적용
    """
    nll = _ensure_float(nll)
    target = _ensure_float(target)

    w = intermittent_weights(
        target, alpha_zero = alpha_zero, alpha_pos = alpha_pos,
        gamma_run = gamma_run, cap = cap
    )
    if use_horizon_decay:
        w = _apply_horizon_decay(w, target, tau_h = tau_h)

    return _weighted_mean(nll, w, eps)