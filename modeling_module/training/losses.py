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
                  - strategy='mix'    → Weighted-Huber(스파이크 가중 + horizon 가중) + AsymMSE 블렌딩
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
                       weight: torch.Tensor, delta: float = 2.0) -> torch.Tensor:
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
                         delta: float = 2.0, up_w: float = 2.0, down_w: float = 1.0,
                         weight: torch.Tensor | None = None) -> torch.Tensor:
        err = pred - y
        abs_e = err.abs()
        huber = torch.where(abs_e <= delta, 0.5 * err * err, delta * (abs_e - 0.5 * delta))
        asym = torch.where(err > 0, torch.full_like(huber, up_w), torch.full_like(huber, down_w))
        loss = asym * huber
        if weight is not None:
            loss = loss * weight
        return loss.mean()

    # --- NEW: horizon 가중(평균 1로 정규화) ---
    def _horizon_weights(self, y: torch.Tensor) -> torch.Tensor:
        """
        cfg.use_horizon_decay=True 이면 [B,H] 형태의 horizon 가중을 반환.
        tau_h<1: 근미래 강조, tau_h>1: 후반부 강조. 평균 1로 정규화.
        """
        use_hd = bool(getattr(self.cfg, "use_horizon_decay", False))
        if not use_hd:
            # 가중 1
            if y.dim() == 3 and y.size(1) == 1:
                return torch.ones_like(y.squeeze(1))
            return torch.ones_like(y)

        tau_h = float(getattr(self.cfg, "tau_h", 1.0))
        if y.dim() == 3 and y.size(1) == 1:
            H = y.size(-1)
            base = y.squeeze(1)
        elif y.dim() == 2:
            H = y.size(-1)
            base = y
        else:
            # 마지막 축을 horizon으로 가정
            H = y.size(-1)
            base = y.view(y.size(0), -1)

        hw = tau_h ** torch.arange(H, device=base.device, dtype=base.dtype)  # 0..H-1
        hw = hw / (hw.mean() + 1e-12)  # 평균 1 정규화
        return hw.unsqueeze(0).expand(base.size(0), H)

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

        # 1) Spike 전략: 'mix' → 블렌딩 (+ horizon 가중 주입)
        if spike_enabled and strategy == "mix":
            y_hat = self._unwrap_point(pred)
            slv = sl

            k = float(getattr(slv, "mad_k", 3.5))
            w_spike = float(getattr(slv, "w_spike", 6.0))
            w_norm = float(getattr(slv, "w_norm", 1.0))
            delta = float(getattr(slv, "huber_delta", getattr(self.cfg, "huber_delta", 2.0)))

            # NEW: asym up/down 둘 다 사용
            up_w = float(getattr(slv, "asym_up_weight", 1.0))
            down_w = float(getattr(slv, "asym_down_weight", 1.0))

            a = float(getattr(slv, "alpha_huber", 0.7))
            b = float(getattr(slv, "beta_asym", 0.3))
            mix_with_baseline = bool(getattr(slv, "mix_with_baseline", False))
            gamma = float(getattr(slv, "gamma_baseline", 0.0))

            # 1) 스파이크 가중
            w_sp = self.make_spike_weight(y, k=k, w_spike=w_spike, w_norm=w_norm)  # [B,H]
            # 2) 호라이즌 가중(평균 1로 정규화)
            hw = self._horizon_weights(y)  # [B,H]

            # 검증에서 가중 끄고 싶으면 cfg.val_use_weights=False
            if is_val and not getattr(self.cfg, "val_use_weights", True):
                w_eff = torch.ones_like(w_sp)
            else:
                w_eff = w_sp * hw

            # NEW: 가중 평균 1로 정규화 (스케일 바이어스 방지)
            w_eff = w_eff / (w_eff.mean() + 1e-12)

            # (옵션) 상한으로 과도한 샘플 폭주 방지
            w_cap = float(getattr(slv, "w_cap", 12.0))
            if w_cap > 0:
                w_eff = torch.clamp(w_eff, max=w_cap)

            # Weighted Huber
            err = y_hat - y
            abs_e = err.abs()
            huber = torch.where(abs_e <= delta, 0.5 * err * err, delta * (abs_e - 0.5 * delta))
            loss_huber = (w_eff * huber).mean()

            # NEW: 비대칭 제곱오차 (언더/오버 모두 가중)
            mse = err.pow(2)
            w_asym = torch.where(err > 0, torch.full_like(mse, up_w), torch.full_like(mse, down_w))
            loss_asym = (w_eff * w_asym * mse).mean()

            loss = a * loss_huber + b * loss_asym

            if mix_with_baseline:
                base = self._compute_point_base(pred, y, is_val=is_val)
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


# --- 이하 유틸(기존 유지) ---
def huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(x, y, delta=delta)

def pinball_loss(yhat: torch.Tensor, y: torch.Tensor, q: float) -> torch.Tensor:
    diff = y - yhat
    return torch.maximum(q * diff, (q - 1) * diff).mean()

def asymmetric_mse(yhat: torch.Tensor, y: torch.Tensor, up_w: float = 2.0, down_w: float = 1.0) -> torch.Tensor:
    diff = yhat - y
    w = torch.where(diff >= 0, torch.as_tensor(up_w, device=diff.device), torch.as_tensor(down_w, device=diff.device))
    return (w * diff.pow(2)).mean()

def detect_spikes(y: torch.Tensor, k: float = 3.5) -> torch.Tensor:
    # MAD-based spike mask on last dim
    med = y.median(dim=-2, keepdim=True).values if y.dim() >= 2 else y.median()
    mad = (y - med).abs().median(dim=-2, keepdim=True).values + 1e-8
    z = (y - med).abs() / mad
    return (z >= k).float()

def spike_friendly_loss(yhat: torch.Tensor, y: torch.Tensor, cfg: TrainingConfig) -> torch.Tensor:
    sc = cfg.spike_loss
    if not sc.enabled:
        return huber_loss(yhat, y, delta=cfg.huber_delta)

    if sc.strategy == 'direct':
        return F.huber_loss(yhat, y, delta=sc.huber_delta) * 0.7 + asymmetric_mse(yhat, y, sc.asym_up_weight, sc.asym_down_weight) * 0.3

    # mix: spike 가중치
    mask = detect_spikes(y, k=sc.mad_k)
    w = mask * sc.w_spike + (1 - mask) * sc.w_norm
    loss_h = F.huber_loss(yhat, y, delta=sc.huber_delta, reduction='none')
    loss_a = (yhat - y).pow(2)
    # 비대칭 가중
    loss_a = torch.where(yhat >= y, loss_a * sc.asym_up_weight, loss_a * sc.asym_down_weight)
    loss = sc.alpha_huber * (w * loss_h).mean() + sc.beta_asym * (w * loss_a).mean()

    if sc.mix_with_baseline:
        loss = (1 - sc.gamma_baseline) * loss + sc.gamma_baseline * huber_loss(yhat, y, delta=cfg.huber_delta)
    return loss

def make_pspa_fn(lambda_pattern: float = 0.20, lambda_scale: float = 0.10,
                 target_channel: int = 0):
    """
    PSPA: Past Scale & Pattern Anchoring
    """

    import torch
    import torch.nn.functional as F
    eps = 1e-8

    def _to_pred_tensor(out):
        """
        out: Tensor | {"y_pred": Tensor} | {"q": [B,Q,H] or [B,Q,H,C]}
        반환: [B,H,1]  (항상 단일 채널로 정규화)
        """
        if isinstance(out, torch.Tensor):
            y = out  # [B,H] or [B,H,C]
        elif isinstance(out, dict):
            if "y_pred" in out:
                y = out["y_pred"]          # [B,H] or [B,H,C]
            elif "q" in out:
                q = out["q"]               # [B,Q,H] or [B,Q,H,C]
                # 중앙(0.5) 선택
                q_idx = q.shape[1] // 2
                y = q[:, q_idx, ...]       # [B,H] or [B,H,C]
            else:
                raise ValueError("Unsupported pred dict keys.")
        else:
            raise ValueError("Unsupported pred type.")

        # 차원 정리 -> [B,H,1]
        if y.dim() == 2:                   # [B,H] -> [B,H,1]
            y = y.unsqueeze(-1)
        elif y.dim() == 3:                 # [B,H,C] -> [B,H,1] (target_channel만 쓰기)
            C = y.size(-1)
            ch = min(max(target_channel, 0), C - 1)
            y = y[..., ch:ch+1]
        else:
            raise ValueError(f"Unsupported pred shape: {y.shape}")
        return y

    def _select_hist_channel(x):
        # x: [B,L,C] -> [B,L,1]
        if x.dim() != 3:
            raise ValueError(f"x must be [B,L,C], got {x.shape}")
        C = x.size(-1)
        ch = min(max(target_channel, 0), C - 1)
        return x[:, :, ch:ch+1]

    def _pspa(x, out, cfg):
        y = _to_pred_tensor(out)       # [B,H,1]
        x_ch = _select_hist_channel(x) # [B,L,1]

        # 차분
        diff_hist = x_ch[:, 1:, :] - x_ch[:, :-1, :]   # [B,L-1,1]
        diff_pred = y[:, 1:, :] - y[:, :-1, :]         # [B,H-1,1]

        # 길이 정렬
        T = min(diff_hist.size(1), diff_pred.size(1))
        if T < 1:
            return y.new_tensor(0.0)

        dh = diff_hist[:, -T:, :]   # [B,T,1]
        dp = diff_pred[:, :T, :]    # [B,T,1]

        # 패턴 앵커(코사인 유사도)
        v1 = dh.reshape(dh.size(0), -1)  # [B, T]
        v2 = dp.reshape(dp.size(0), -1)  # [B, T]
        cos = F.cosine_similarity(v1, v2, dim=1)  # [B]
        L_pattern = (1.0 - cos).mean()

        # 스케일 앵커
        hist_seg = x_ch[:, -T:, :]     # [B,T,1]
        pred_seg = y[:, :T, :]         # [B,T,1]
        scale_hist = hist_seg.abs().mean(dim=(1, 2)) + eps
        scale_pred = pred_seg.abs().mean(dim=(1, 2)) + eps
        L_scale = ((scale_pred / scale_hist) - 1.0).abs().mean()

        return lambda_pattern * L_pattern + lambda_scale * L_scale

    return _pspa

