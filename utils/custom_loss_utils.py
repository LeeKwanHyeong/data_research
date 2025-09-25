# utils/custom_loss_utils.py

from typing import Optional, Tuple, Iterable, Dict
import torch
import torch.nn.functional as F

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

EPS = 1e-12


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_floating_point() else x.float()


def _safe_weighted_mean(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x, w: 동일 shape(브로드캐스트 이후). 항상 finite tensor 반환.
    w가 None이면 단순 평균. 가중합이 0이면 EPS 바닥을 사용.
    """
    if w is None:
        return x.mean()
    x = _as_float_tensor(x)
    w = _as_float_tensor(w).to(device=x.device)
    num = (x * w).sum()
    den = w.sum()
    return num / (den + EPS)


def _ensure_w_shape(w: Optional[torch.Tensor], like: torch.Tensor) -> Optional[torch.Tensor]:
    """
    like: y 또는 pred, shape (B,H) 또는 (B,Q,H)
    w: (B,H) 또는 브로드캐스트 가능 텐서.
    (B,Q,H)에서 w를 (B,1,H)로 확장해 Q축 브로드캐스트가 되도록 맞춤.
    """
    if w is None:
        return None
    like = _as_float_tensor(like)
    w = _as_float_tensor(w).to(device=like.device)
    if like.dim() == 3:  # (B,Q,H)
        if w.dim() == 2:
            w = w.unsqueeze(1)  # (B,1,H)
    return w


def _apply_horizon_decay(w: torch.Tensor, target_like: torch.Tensor, tau_h: float = 24.0) -> torch.Tensor:
    """
    w, target_like: (B,H) 가정(필요시 상위에서 맞춰서 전달)
    tau_h가 클수록 완만, 작을수록 급격 감쇠.
    """
    target_like = _as_float_tensor(target_like)
    w = _as_float_tensor(w).to(device=target_like.device)
    B, H = target_like.shape
    h = torch.arange(H, device=target_like.device, dtype=target_like.dtype).view(1, -1)
    decay = torch.exp(-h / max(float(tau_h), 1.0))
    return w * decay


def _pinball_elem(diff: torch.Tensor, q: float) -> torch.Tensor:
    """
    element-wise pinball for a given quantile q
    diff = target - pred
    """
    q_t = torch.as_tensor(q, dtype=diff.dtype, device=diff.device)
    return torch.maximum(q_t * diff, (q_t - 1.0) * diff)


# =====================================================
# Intermittent weights
# =====================================================

def _zero_runlength(target: torch.Tensor) -> torch.Tensor:
    """
    target: (B,H) (>=0 가정)
    return: (B,H) 각 시점까지 연속 0의 길이(해당 시점 포함)
    """
    target = _as_float_tensor(target)
    B, H = target.shape
    r = torch.zeros_like(target)
    for t in range(H):
        if t == 0:
            r[:, t] = (target[:, t] == 0).to(target.dtype)
        else:
            r[:, t] = torch.where(target[:, t] == 0, r[:, t - 1] + 1.0, torch.zeros_like(target[:, t]))
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
    연속 0-run 길이에 기반한 간헐 가중 생성.
    balanced=False: 기본형
    balanced=True : zero/positive 각각 평균 1이 되도록 정규화 (clip_run으로 run 영향 상한)
    """
    target = _as_float_tensor(target)
    B, H = target.shape

    run = _zero_runlength(target)
    norm_run = run / max(1, H)
    if balanced:
        norm_run = norm_run.clamp(max=clip_run)

    is_zero = (target == 0).to(target.dtype)
    is_pos = 1.0 - is_zero

    w_zero = alpha_zero * is_zero * (1.0 + gamma_run * norm_run)
    w_pos = alpha_pos * is_pos

    if balanced:
        # zero/pos 각각 평균 1이 되도록 스케일
        m_zero = (w_zero.sum() / (is_zero.sum() + eps)).clamp_min(eps)
        m_pos = (w_pos.sum() / (is_pos.sum() + eps)).clamp_min(eps)
        w = (w_zero / m_zero) + (w_pos / m_pos)
    else:
        w = w_zero + w_pos
        if cap is not None:
            w = torch.clamp(w, max=float(cap))

    return w


def intermittent_weights(
    target: torch.Tensor,
    alpha_zero: float = 1.2,
    alpha_pos: float = 1.0,
    gamma_run: float = 0.6,
    cap: Optional[float] = None,
) -> torch.Tensor:
    """
    기본형 간헐 가중
    w = α_zero·I[y=0]·(1 + γ·runlen_norm) + α_pos·I[y>0]
    cap: 너무 큰 가중 폭주 방지 상한
    """
    return _intermittent_weights_core(
        target, alpha_zero, alpha_pos, gamma_run, cap, balanced=False
    )


def intermittent_weights_balanced(
    target: torch.Tensor,
    alpha_zero: float = 1.2,
    alpha_pos: float = 1.0,
    gamma_run: float = 0.6,
    clip_run: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    balanced 버전: zero/pos 각 집합 평균이 1이 되도록 정규화.
    연속 0-run 영향은 clip_run 이하로 제한.
    """
    return _intermittent_weights_core(
        target, alpha_zero, alpha_pos, gamma_run,
        cap=None, balanced=True, clip_run=clip_run, eps=eps
    )

# =====================================================
# Pinball losses
# =====================================================

def pinball_plain(
    pred_q: torch.Tensor,
    y: torch.Tensor,
    quantiles: Iterable[float] = (0.1, 0.5, 0.9)
) -> torch.Tensor:
    """
    pred_q: (B,Q,H), y: (B,H)
    모든 분위수에 대한 pinball 평균
    """
    y = _as_float_tensor(y)
    pred_q = _as_float_tensor(pred_q).to(device=y.device)
    diff = y.unsqueeze(1) - pred_q  # (B,Q,H)
    losses = []
    for i, q in enumerate(quantiles):
        losses.append(_pinball_elem(diff[:, i], float(q)).mean())
    return sum(losses) / max(1, len(losses))


def pinball_loss_weighted(
    pred_q: torch.Tensor,
    target: torch.Tensor,
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    pred_q: (B,Q,H), target: (B,H)
    weights: (B,H) 또는 (B,1,H)로 브로드캐스트 가능
    """
    pred_q = _as_float_tensor(pred_q)
    target = _as_float_tensor(target).to(device=pred_q.device)

    diff = target.unsqueeze(1) - pred_q  # (B,Q,H)
    if weights is None:
        # 비가중 pinball
        losses = [_pinball_elem(diff[:, i], float(q)).mean() for i, q in enumerate(quantiles)]
        return sum(losses) / max(1, len(quantiles))

    # 가중 pinball (Q축 동일 가중)
    w = _ensure_w_shape(weights, pred_q)  # (B,1,H)
    losses = []
    for i, q in enumerate(quantiles):
        e = _pinball_elem(diff[:, i], float(q))  # (B,H)
        losses.append(_safe_weighted_mean(e, w.squeeze(1)))
    return sum(losses) / max(1, len(quantiles))


def pinball_loss_weighted_masked(
    pred_q: torch.Tensor,
    target: torch.Tensor,
    quantiles: Tuple[float, ...],
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    하위 분위수(q<0.5)에만 가중 적용, 나머지는 비가중 평균.
    pred_q: (B,Q,H), target: (B,H), weights: (B,H) or None
    """
    pred_q = _as_float_tensor(pred_q)
    target = _as_float_tensor(target).to(device=pred_q.device)

    diff = target.unsqueeze(1) - pred_q  # (B,Q,H)
    w = _ensure_w_shape(weights, pred_q)  # (B,1,H) or None
    losses = []
    for i, q in enumerate(quantiles):
        e = _pinball_elem(diff[:, i], float(q))  # (B,H)
        if (w is not None) and (q < 0.5):
            losses.append(_safe_weighted_mean(e, w.squeeze(1)))
        else:
            losses.append(e.mean())
    return sum(losses) / max(1, len(quantiles))

# =====================================================
# Horizon decay weights
# =====================================================

def horizon_decay_weights(target_like: torch.Tensor, tau_h: float = 24.0) -> torch.Tensor:
    """
    target_like: (H,) 또는 (B,H)
    tau_h: 클수록 완만, 작을수록 급격 감쇠
    """
    target_like = _as_float_tensor(target_like)
    if target_like.dim() == 1:
        H = target_like.size(0)
        device, dtype = target_like.device, target_like.dtype
        h_idx = torch.arange(H, device=device, dtype=dtype)
        return torch.exp(-h_idx / max(float(tau_h), 1.0))  # (H,)
    elif target_like.dim() == 2:
        B, H = target_like.shape
        device, dtype = target_like.device, target_like.dtype
        h_idx = torch.arange(H, device=device, dtype=dtype)
        return torch.exp(-h_idx / max(float(tau_h), 1.0)).unsqueeze(0).expand(B, H)  # (B,H)
    else:
        raise ValueError("target_like must be (H,) or (B,H)")

# =====================================================
# Newsvendor q*
# =====================================================

def newsvendor_q_star(Cu: float, Co: float) -> float:
    return float(Cu) / max(float(Cu) + float(Co), EPS)

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
) -> torch.Tensor:
    """
    간헐 가중 + (옵션) 호라이즌 감쇠를 멀티-분위수 pinball에 적용
    """
    target = _as_float_tensor(target)
    w = intermittent_weights(target, alpha_zero=alpha_zero, alpha_pos=alpha_pos, gamma_run=gamma_run, cap=cap)
    if use_horizon_decay:
        w = _apply_horizon_decay(w, target, tau_h=tau_h)
    return pinball_loss_weighted(pred_q, target, quantiles=quantiles, weights=w)


def intermittent_point_loss(
    y_hat: torch.Tensor,
    target: torch.Tensor,
    mode: str = "mae",      # 'mae' | 'mse' | 'huber' | 'pinball'
    tau: float = 0.5,       # pinball q
    delta: float = 1.0,     # huber delta
    alpha_zero: float = 1.2,
    alpha_pos: float = 1.0,
    gamma_run: float = 0.6,
    cap: Optional[float] = None,
    use_horizon_decay: bool = False,
    tau_h: float = 24.0,
) -> torch.Tensor:
    """
    점추정 모델(Titan 등)에 간헐 가중을 적용한 손실.
    - 간헐 가중 OFF 형태를 원하면 alpha_zero=0.0, alpha_pos=1.0로 호출.
    """
    y_hat = _as_float_tensor(y_hat)
    target = _as_float_tensor(target).to(device=y_hat.device)

    # 기본(비가중) 손실 성분
    if mode == "mae":
        e = (y_hat - target).abs()
    elif mode == "mse":
        e = (y_hat - target).pow(2)
    elif mode == "huber":
        e = F.huber_loss(y_hat, target, delta=delta, reduction="none")
    elif mode == "pinball":
        diff = target - y_hat
        e = _pinball_elem(diff, float(tau))
    else:
        raise ValueError("mode must be one of ['mae', 'mse', 'huber', 'pinball']")

    # 간헐 가중 OFF 옵션(기본 평균)
    if alpha_zero == 0.0 and alpha_pos == 1.0:
        return e.mean()

    # 간헐 가중
    w = intermittent_weights(target, alpha_zero=alpha_zero, alpha_pos=alpha_pos, gamma_run=gamma_run, cap=cap)
    if use_horizon_decay:
        w = _apply_horizon_decay(w, target, tau_h=tau_h)

    return _safe_weighted_mean(e, w)


def intermittent_nll_loss(
    nll: torch.Tensor,
    target: torch.Tensor,
    alpha_zero: float = 1.2,
    alpha_pos: float = 1.0,
    gamma_run: float = 0.6,
    cap: Optional[float] = None,
    use_horizon_decay: bool = False,
    tau_h: float = 24.0,
) -> torch.Tensor:
    """
    DeepAR/Tweedie/ZI-NB 등 분포형 모델의 NLL에 간헐 가중 적용.
    """
    nll = _as_float_tensor(nll)
    target = _as_float_tensor(target).to(device=nll.device)

    w = intermittent_weights(target, alpha_zero=alpha_zero, alpha_pos=alpha_pos, gamma_run=gamma_run, cap=cap)
    if use_horizon_decay:
        w = _apply_horizon_decay(w, target, tau_h=tau_h)

    return _safe_weighted_mean(nll, w)
