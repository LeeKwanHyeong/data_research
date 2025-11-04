import torch
import torch.nn.functional as F
from typing import Optional

try:
    from modeling_module.training.config import TrainingConfig
except Exception:
    TrainingConfig = object

try:
    from modeling_module.utils.custom_loss_utils import (
        intermittent_weights_balanced,
        intermittent_point_loss,
        newsvendor_q_star,
        pinball_plain,
        pinball_loss_weighted_masked
    )
except Exception:
    # --- 최소 폴백 ---
    def newsvendor_q_star(Cu: float, Co: float) -> float:
        Cu = float(Cu); Co = float(Co)
        return Cu / (Cu + Co + 1e-12)

    def pinball_plain(pred, y, quantiles):
        if y.dim() == 3:  # [B,1,H]
            y = y.squeeze(1)
        B, Q, H = pred.shape
        loss = 0.0
        for i, q in enumerate(quantiles):
            diff = y - pred[:, i, :]
            loss_q = torch.maximum(q * diff, (q - 1.0) * diff).mean()
            loss += loss_q
        return loss / len(quantiles)

    def pinball_loss_weighted_masked(pred, y, quantiles, weights=None):
        if y.dim() == 3: y = y.squeeze(1)
        B, Q, H = pred.shape
        total = 0.0
        for i, q in enumerate(quantiles):
            diff = y - pred[:, i, :]
            loss_q = torch.maximum(q * diff, (q - 1.0) * diff)
            if weights is not None:
                loss_q = loss_q * weights
            total += loss_q.mean()
        return total / len(quantiles)

    def intermittent_point_loss(pred, y, *, mode, tau, delta, **_):
        if mode == "mae":   return (pred - y).abs().mean()
        if mode == "mse":   return F.mse_loss(pred, y)
        if mode == "huber": return F.huber_loss(pred, y, delta=delta)
        if mode == "pinball":
            diff = y - pred
            return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()
        return (pred - y).abs().mean()

    def intermittent_weights_balanced(y, alpha_zero, alpha_pos, gamma_run, clip_run=0.5):
        is_zero = (y <= 0)
        return torch.where(is_zero, torch.full_like(y, alpha_zero), torch.full_like(y, alpha_pos))


class LossComputer:
    """
    단일 엔트리 포인트 compute()로 모든 손실 모드를 처리:
      - Quantile: pinball (가중/마스크 가능)
      - Point   : mae / mse / huber / pinball(q*) / huber_asym(비대칭 Huber)
      - Spike   : cfg.spike_loss.enabled=True 일 때
                  - strategy='mix'    → Weighted-Huber(스파이크 가중) + AsymMSE 블렌딩
                  - strategy='direct' → point_loss='huber_asym' 단독 사용

    차이:
      - 'mix'    : 피크 민감도(Weighted-Huber) + 과대예측 벌점(AsymMSE)을 동시에 반영(블렌딩)
      - 'direct' : 전체 구간에 일관된 비대칭 비용(단일 huber_asym), 단순/안정
    """
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg

    # ---------- helpers ----------
    @staticmethod
    def _unwrap_point(pred):
        """dict/quantile를 포인트 텐서 [B,H]로 정규화"""
        p = pred
        if isinstance(p, dict):
            if "point" in p:
                p = p["point"]
            elif "q" in p:
                q = p["q"]
                if q.dim() == 3 and q.size(-1) >= 3:
                    p = q[..., 1]  # q50
                else:
                    p = q[..., 0]
            else:
                p = next(iter(p.values()))
        if p.dim() == 3 and p.size(-1) == 1:
            p = p.squeeze(-1)
        return p

    @staticmethod
    def _as_tensor(x, like: torch.Tensor):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x, dtype=like.dtype, device=like.device)

    def _q_star(self) -> float:
        if getattr(self.cfg, "use_cost_q_star", False) and self.cfg.point_loss == 'pinball':
            return float(newsvendor_q_star(self.cfg.Cu, self.cfg.Co))
        return float(getattr(self.cfg, "q_star", 0.5))

    # ---------- spike weights & primitive losses ----------
    @staticmethod
    def make_spike_weight(y_hist: torch.Tensor, k: float = 3.5,
                          w_spike: float = 6.0, w_norm: float = 1.0) -> torch.Tensor:
        if y_hist.dim() == 3 and y_hist.size(1) == 1:
            y_hist = y_hist.squeeze(1)
        med = torch.median(y_hist, dim=1, keepdim=True).values
        mad = torch.median(torch.abs(y_hist - med), dim=1, keepdim=True).values + 1e-6
        z = (y_hist - med) / mad
        spike = (z > k).float()
        return torch.where(spike > 0, torch.full_like(y_hist, w_spike), torch.full_like(y_hist, w_norm))

    @staticmethod
    def weighted_huber(y_hat: torch.Tensor, y_true: torch.Tensor,
                       weight: torch.Tensor, delta: float = 5.0) -> torch.Tensor:
        err = y_hat - y_true
        abs_e = err.abs()
        huber = torch.where(abs_e <= delta, 0.5 * err * err, delta * (abs_e - 0.5 * delta))
        return (weight * huber).mean()

    @staticmethod
    def asymmetric_mse(y_hat: torch.Tensor, y_true: torch.Tensor, up_w: float = 2.0) -> torch.Tensor:
        e = y_hat - y_true
        w = torch.where(e < 0, torch.ones_like(e), torch.full_like(e, up_w))
        return (w * e.pow(2)).mean()

    @staticmethod
    def huber_asymmetric(pred: torch.Tensor, y: torch.Tensor, *,
                         delta: float = 5.0, up_w: float = 2.0, down_w: float = 1.0,
                         weight: torch.Tensor | None = None) -> torch.Tensor:
        err = pred - y
        abs_e = err.abs()
        huber = torch.where(abs_e <= delta, 0.5 * err * err, delta * (abs_e - 0.5 * delta))
        asym = torch.where(err > 0, torch.full_like(huber, up_w), torch.full_like(huber, down_w))
        loss = asym * huber
        if weight is not None:
            loss = loss * weight
        return loss.mean()

    # ---------- single entry ----------
    def compute(self, pred: torch.Tensor | dict, y: torch.Tensor, *, is_val: bool) -> torch.Tensor:
        """
        단일 엔트리. cfg.spike_loss.enabled/strategy에 따라 분기.
        """
        sl = getattr(self.cfg, "spike_loss", None)
        spike_enabled = bool(getattr(sl, "enabled", False)) if sl else False
        strategy = (getattr(sl, "strategy", "mix") if sl else "off")  # 'mix' | 'direct' | 'off'

        # 0) Quantile 경로 (항상 최우선)
        mode_cfg = getattr(self.cfg, "loss_mode", "auto")
        if mode_cfg == "auto":
            if (torch.is_tensor(pred) and pred.dim() == 3 and pred.size(1) > 1) or (isinstance(pred, dict) and "q" in pred):
                mode = "quantile"
            else:
                mode = "point"
        else:
            mode = mode_cfg

        if mode == "quantile":
            if is_val and not getattr(self.cfg, "val_use_weights", True):
                return pinball_plain(pred, y, getattr(self.cfg, "quantiles", [0.1, 0.5, 0.9]))
            weights: Optional[torch.Tensor] = None
            if getattr(self.cfg, "use_intermittent", False):
                weights = intermittent_weights_balanced(
                    y,
                    alpha_zero=getattr(self.cfg, "alpha_zero", 1.0),
                    alpha_pos=getattr(self.cfg, "alpha_pos", 1.0),
                    gamma_run=getattr(self.cfg, "gamma_run", 0.0),
                    clip_run=0.5,
                )
                if torch.sum(weights) == 0:
                    weights = torch.ones_like(y) * 1e-6
            return pinball_loss_weighted_masked(
                pred, y, getattr(self.cfg, "quantiles", [0.1, 0.5, 0.9]), weights
            )

        # 1) Spike 전략: 'mix' → 블렌딩
        if spike_enabled and strategy == "mix":
            y_hat = self._unwrap_point(pred)
            slv = sl  # dict 또는 네임스페이스 가정
            k       = float(getattr(slv, "mad_k", 3.5))
            w_spike = float(getattr(slv, "w_spike", 6.0))
            w_norm  = float(getattr(slv, "w_norm", 1.0))
            delta   = float(getattr(slv, "huber_delta", getattr(self.cfg, "huber_delta", 5.0)))
            up_w    = float(getattr(slv, "asym_up_weight", 2.0))
            a       = float(getattr(slv, "alpha_huber", 0.7))
            b       = float(getattr(slv, "beta_asym", 0.3))
            mix_with_baseline = bool(getattr(slv, "mix_with_baseline", False))
            gamma   = float(getattr(slv, "gamma_baseline", 0.2))

            w = self.make_spike_weight(y, k=k, w_spike=w_spike, w_norm=w_norm)
            loss_huber = self.weighted_huber(y_hat, y, w, delta=delta)
            loss_asym  = self.asymmetric_mse(y_hat, y, up_w=up_w)
            loss = a * loss_huber + b * loss_asym

            if mix_with_baseline:
                base = self._compute_point_base(pred, y, is_val=is_val)
                base = self._as_tensor(base, y_hat) or (y_hat - y).abs().mean()
                loss = loss + gamma * base

            return loss

        # 2) Spike 전략: 'direct' → point_loss='huber_asym' 경로
        if spike_enabled and strategy == "direct" and getattr(self.cfg, "point_loss", "mae") == "huber_asym":
            y_hat = self._unwrap_point(pred)
            w = None
            if not (is_val and not getattr(self.cfg, "val_use_weights", True)):
                if getattr(self.cfg, "use_intermittent", False):
                    w = intermittent_weights_balanced(
                        y,
                        alpha_zero=getattr(self.cfg, "alpha_zero", 1.0),
                        alpha_pos=getattr(self.cfg, "alpha_pos", 1.0),
                        gamma_run=getattr(self.cfg, "gamma_run", 0.0),
                        clip_run=0.5,
                    )
            return self.huber_asymmetric(
                y_hat, y,
                delta=getattr(sl, "huber_delta", getattr(self.cfg, "huber_delta", 5.0)),
                up_w=getattr(sl, "asym_up_weight", 2.0),
                down_w=getattr(sl, "asym_down_weight", 1.0),
                weight=w,
            )

        # 3) 일반 point 경로
        return self._compute_point_base(pred, y, is_val=is_val)

    # --- 내부: 기본 point 손실 ---
    def _compute_point_base(self, pred, y, *, is_val: bool) -> torch.Tensor:
        y_hat = self._unwrap_point(pred)
        pl = getattr(self.cfg, "point_loss", "mae")

        # 검증에서 가중 끄기
        if is_val and not getattr(self.cfg, "val_use_weights", True):
            if pl == 'mae':   return (y_hat - y).abs().mean()
            if pl == 'mse':   return F.mse_loss(y_hat, y)
            if pl == 'huber': return F.huber_loss(y_hat, y, delta=getattr(self.cfg, "huber_delta", 5.0))
            if pl == 'pinball':
                q = self._q_star()
                diff = y - y_hat
                return torch.maximum(q * diff, (q - 1.0) * diff).mean()

        # 가중/마스크 있는 간헐수요 포인트 손실
        return intermittent_point_loss(
            y_hat, y,
            mode=pl,
            tau=self._q_star(),
            delta=getattr(self.cfg, "huber_delta", 5.0),
            alpha_zero=getattr(self.cfg, "alpha_zero", 0.0) if getattr(self.cfg, "use_intermittent", False) else 0.0,
            gamma_run=getattr(self.cfg, "gamma_run", 0.0),
            cap=getattr(self.cfg, "cap", None),
            use_horizon_decay=getattr(self.cfg, "use_horizon_decay", False),
            tau_h=getattr(self.cfg, "tau_h", 1.0),
        )