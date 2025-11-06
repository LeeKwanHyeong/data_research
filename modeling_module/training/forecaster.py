# forecaster.py
# -------------------------------------------------------
# Forecaster for models that already output RAW predictions.
# (Model does RevIN.norm at input and denorm at output.)
# -------------------------------------------------------
import torch
from typing import Optional, Callable

DEBUG_FCAST = True


# -------------------- Utilities --------------------
def _tvar(t: torch.Tensor) -> float:
    if t.dim() >= 2:
        t2 = t.reshape(t.size(0), t.size(1), -1).mean(-1)
        return t2.var(dim=1).mean().item()
    return float('nan')


def _tfirst5(t: torch.Tensor) -> str:
    if t.dim() == 1:
        x = t[:5].detach().cpu().tolist()
    elif t.dim() >= 2:
        x = t[0, :min(5, t.size(1))]
        if x.dim() > 1:
            x = x[..., 0]
        x = x.detach().cpu().tolist()
    else:
        x = []
    return "[" + ", ".join(f"{v:.6g}" for v in x) + "]"


def make_calendar_exo(start_idx: int, H: int, period: int = 52, device: str | torch.device = 'cpu') -> torch.Tensor:
    t = torch.arange(start_idx, start_idx + H, device=device, dtype=torch.float32)
    exo = torch.stack([torch.sin(2 * torch.pi * t / period),
                       torch.cos(2 * torch.pi * t / period)], dim=-1)  # (H, 2)
    return exo


def _prepare_next_input(
    x_raw: torch.Tensor,
    y_step_raw: torch.Tensor,
    *,
    target_channel: int = 0,
    fill_mode: str = 'copy_last',   # {'copy_last','zeros'}
) -> torch.Tensor:
    """
    x_raw: [B, L, C]  (RAW space)
    y_step_raw: [B]   (RAW one-step prediction)
    """
    assert x_raw.dim() == 3, f"x must be [B, L, C], got {x_raw.shape}"
    B, L, C = x_raw.shape
    y_step_raw = y_step_raw.reshape(B, 1, 1)  # -> [B,1,1]

    if C == 1:
        new_token = y_step_raw
    else:
        last = x_raw[:, -1:, :].clone()
        new_token = torch.zeros_like(last) if fill_mode == 'zeros' else last
        new_token[:, 0, target_channel] = y_step_raw[:, 0, 0]

    x_next = torch.cat([x_raw[:, 1:, :], new_token], dim=1)
    return x_next


# ----- guards (RAW space) -----
def _winsorize_clamp_raw(
    hist_raw: torch.Tensor,     # [B, L]
    y_step_raw: torch.Tensor,   # [B]
    *,
    nonneg: bool = True,
    clip_q: tuple[float, float] = (0.05, 0.95),
    clip_mul: float = 2.0,
    max_growth: float = 1.2
) -> torch.Tensor:
    hist = hist_raw.float()
    y    = y_step_raw.float()

    B, L = hist.shape
    last = hist[:, -1]

    hist_safe = torch.where(torch.isfinite(hist), hist, last.unsqueeze(1))
    q_lo = torch.quantile(hist_safe, clip_q[0], dim=1)  # [B]
    q_hi = torch.quantile(hist_safe, clip_q[1], dim=1)  # [B]

    min_cap = torch.zeros_like(q_lo) if nonneg else q_lo
    cap_quant = q_hi * clip_mul
    cap_growth = torch.where(last > 0, last * max_growth, cap_quant)
    max_cap = torch.minimum(cap_quant, cap_growth)

    y = torch.where(torch.isnan(y), last, y)
    y = torch.where(torch.isposinf(y), max_cap, y)
    y = torch.where(torch.isneginf(y), min_cap, y)

    y = torch.clamp(y, min=min_cap, max=max_cap)
    return y


def _dampen_to_last_raw(last_raw: torch.Tensor, y_step_raw: torch.Tensor, *, damp: float = 0.3) -> torch.Tensor:
    if damp <= 0.0:
        return y_step_raw
    return (1.0 - damp) * last_raw + damp * y_step_raw


def _guard_multiplicative_raw(
    last_raw: torch.Tensor,     # [B]
    y_raw: torch.Tensor,        # [B]
    *,
    max_step_up: float = 0.05,
    max_step_down: float = 0.40
) -> torch.Tensor:
    eps = 1e-6
    last_safe = torch.clamp(last_raw, min=eps)
    y_safe = torch.clamp(y_raw, min=eps)

    ratio = y_safe / last_safe
    log_ratio = torch.log(ratio)

    log_min = torch.log(torch.tensor(1.0 - max_step_down, device=last_raw.device))
    log_max = torch.log(torch.tensor(1.0 + max_step_up, device=last_raw.device))

    log_ratio = torch.clamp(log_ratio, min=log_min, max=log_max)
    y_guard = last_safe * torch.exp(log_ratio)
    return y_guard


# -------------------- Forecaster (RAW) --------------------
class DMSForecaster:
    """
    DMS(Direct Multi-Step) + IMS(autoregressive extension) forecaster
    for models that already return **RAW-space** predictions.

    - 입력/슬라이딩 모두 RAW 유지
    - 모델 호출 시 future_exo_cb가 있다면 (B,H,exo)로 전달
    - 가드/윈저/댐핑은 RAW 히스토리 기준
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        target_channel: int = 0,
        fill_mode: str = "copy_last",
        lmm_mode: Optional[str] = None,
        predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ttm: Optional[object] = None,
        future_exo_cb: Optional[Callable[[int, int], torch.Tensor]] = lambda s, h: make_calendar_exo(s, h, period=52),
    ):
        self.model = model
        self.target_channel = target_channel
        self.fill_mode = fill_mode
        self.lmm_mode = lmm_mode
        self.predict_fn = predict_fn
        self.ttm = ttm
        self.future_exo_cb = future_exo_cb
        self.global_t0 = 0  # for exo index

    # ---------- internal helpers ----------
    def _unwrap_output(self, y_full):
        if isinstance(y_full, dict):
            if "point" in y_full:
                y_full = y_full["point"]
            elif "q" in y_full:
                q = y_full["q"]
                if q.dim() == 3 and q.size(-1) >= 3:
                    y_full = q[..., 1]  # q50
                else:
                    y_full = q[..., 0]  # first channel
            else:
                k = next(iter(y_full))
                y_full = y_full[k]
        return y_full

    def _get_h_hint(self) -> int:
        return int(getattr(self.model, "horizon", getattr(self.model, "output_horizon", 0)) or 0)

    def _normalize_by_horizon(self, y_full, B: int, H_hint: Optional[int] = None) -> torch.Tensor:
        """Just fix shape to [B,H] (NO normalization!)."""
        y_full = self._unwrap_output(y_full)

        if y_full.dim() == 1:
            return y_full.view(B, -1)
        if y_full.dim() == 2:
            return y_full  # [B,H]
        if y_full.dim() == 3:
            d1, d2 = y_full.size(1), y_full.size(2)

            if H_hint is not None:
                if d1 == H_hint and d2 != H_hint:
                    return y_full[:, :, 0]
                if d2 == H_hint and d1 != H_hint:
                    return y_full[:, 0, :]
                if d1 == H_hint and d2 == H_hint:
                    return y_full[:, :, 0]

            if d2 in (1, 3):
                return y_full[:, :, 1] if d2 == 3 else y_full[:, :, 0]
            return y_full[:, 0, :]

        return y_full.reshape(B, -1)

    def _call_model(self, x_raw: torch.Tensor, B: int, future_exo: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Call model; model returns RAW forecasts."""
        H_hint = self._get_h_hint()
        if self.predict_fn is not None:
            y = self.predict_fn(x_raw)
            return self._normalize_by_horizon(y, B, H_hint)

        # try with/without extra args
        tries = []
        if future_exo is not None:
            tries += [dict(future_exo=future_exo, mode=(self.lmm_mode or "eval")),
                      dict(future_exo=future_exo)]
        tries += [dict(mode=(self.lmm_mode or "eval")), dict()]

        for args in tries:
            try:
                y = self.model(x_raw, **args)
                return self._normalize_by_horizon(y, B, H_hint)
            except TypeError:
                continue

        # last resort
        y = self.model(x_raw)
        return self._normalize_by_horizon(y, B, H_hint)

    # ---------- public ----------
    @torch.no_grad()
    def forecast_DMS_to_IMS(
        self,
        x_init: Optional[torch.Tensor] = None,   # preferred name
        *,
        x: Optional[torch.Tensor] = None,        # alias
        horizon: Optional[int] = None,
        device: Optional[torch.device] = None,
        extend: str = "ims",                    # {'ims','error'}
        context_policy: str = "per_step",           # {'once','per_step','off'}
        y_true: Optional[torch.Tensor] = None,  # (RAW) TF target
        teacher_forcing_ratio: float = 0.0,

        # RAW-space stabilization toggles
        use_winsor: bool = False,
        use_multi_guard: bool = False,
        use_dampen: bool = False,
        winsor_q: tuple = (0.05, 0.95),
        winsor_mul: float = 2.0,
        winsor_growth: float = 1.2,
        max_step_up: float = 0.05,
        max_step_down: float = 0.40,
        damp: float = 0.30,
    ) -> torch.Tensor:
        """
        Returns RAW-space y_hat: [B,H]
        """
        # ---- input unify ----
        x_in = x_init if x_init is not None else x
        if x_in is None:
            raise TypeError("forecast_DMS_to_IMS requires 'x_init' (preferred) or 'x'.")

        was_training = self.model.training
        self.model.eval()

        device = device or next(self.model.parameters()).device
        x_raw = x_in.to(device).float().clone()
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(-1)  # [B,L] -> [B,L,1]
        B, L, C = x_raw.shape

        # --- TTM context (optional) ---
        if (self.ttm is not None) and (context_policy in ("once", "per_step")):
            if context_policy == "once":
                # if your TTM expects encoded features, adapt here
                self.ttm.add_context(x_raw)

        # --- Hm estimation ---
        def _probe_hm_safe() -> int:
            try:
                return self._call_model(x_raw, B).size(1)
            except Exception:
                H_guess = self._get_h_hint() or 120
                exo = None
                if self.future_exo_cb is not None:
                    exo = self.future_exo_cb(self.global_t0, H_guess).to(x_raw.device).unsqueeze(0).expand(B, -1, -1)
                return self._call_model(x_raw, B, exo).size(1)

        Hm = _probe_hm_safe()
        H = int(horizon) if horizon is not None else Hm

        # --- DMS block (RAW) ---
        def _call_with_exo(xr: torch.Tensor, need: int, step_offset: int):
            exo = None
            if self.future_exo_cb is not None:
                t0 = self.global_t0 + step_offset
                exo = self.future_exo_cb(t0, need).to(xr.device)  # (H, exo)
                exo = exo.unsqueeze(0).expand(B, -1, -1)
            return self._call_model(xr, B, future_exo=exo)  # [B,need] RAW

        y_block_raw = _call_with_exo(x_raw, Hm, 0)  # [B,Hm] RAW

        if DEBUG_FCAST:
            print(f"[FCAST-DBG] DMS block: Hm={y_block_raw.size(1)}, "
                  f"var(Hm)={_tvar(y_block_raw):.6g}, first5={_tfirst5(y_block_raw)}")

        outputs = []
        use_tf = (y_true is not None) and (teacher_forcing_ratio > 0.0)
        if use_tf:
            y_true = y_true.to(device).float()  # RAW target

        # main part: min(Hm, H)
        use_len = min(Hm, H)
        for t in range(use_len):
            if (self.ttm is not None) and (context_policy == "per_step"):
                self.ttm.add_context(x_raw)

            # model one-step (from the DMS block)
            y_step_raw = y_block_raw[:, t]  # [B] RAW

            # guards in RAW space
            hist_raw = x_raw[:, :, self.target_channel]
            last_raw = hist_raw[:, -1]
            y_adj = y_step_raw

            if use_winsor:
                y_adj = _winsorize_clamp_raw(hist_raw, y_adj,
                                             nonneg=True, clip_q=winsor_q,
                                             clip_mul=winsor_mul, max_growth=winsor_growth)
            if use_multi_guard:
                y_adj = _guard_multiplicative_raw(last_raw, y_adj,
                                                  max_step_up=max_step_up, max_step_down=max_step_down)
            if use_dampen:
                y_adj = _dampen_to_last_raw(last_raw, y_adj, damp=damp)

            outputs.append(y_adj.unsqueeze(1))  # [B,1]

            # slide window with RAW value
            x_raw = _prepare_next_input(x_raw, y_adj,
                                        target_channel=self.target_channel,
                                        fill_mode=self.fill_mode)

        # IMS extension
        if H > Hm:
            if extend not in ("ims", "error"):
                raise ValueError("extend must be 'ims' or 'error'")
            if extend == "error":
                raise ValueError(f"horizon ({H}) > model_output ({Hm}). "
                                 f"Set extend='ims' to extend autoregressively.")

            remaining = H - use_len
            for t in range(remaining):
                if (self.ttm is not None) and (context_policy == "per_step"):
                    self.ttm.add_context(x_raw)

                # call model with sliding window; take the very next step
                y_full_raw = _call_with_exo(x_raw, Hm, step_offset=(use_len + t))  # [B,Hm] RAW
                y_step_raw = y_full_raw[:, 0]

                # guards in RAW
                hist_raw = x_raw[:, :, self.target_channel]
                last_raw = hist_raw[:, -1]
                y_adj = y_step_raw
                if use_winsor:
                    y_adj = _winsorize_clamp_raw(hist_raw, y_adj,
                                                 nonneg=True, clip_q=winsor_q,
                                                 clip_mul=winsor_mul, max_growth=winsor_growth)
                if use_multi_guard:
                    y_adj = _guard_multiplicative_raw(last_raw, y_adj,
                                                      max_step_up=max_step_up, max_step_down=max_step_down)
                if use_dampen:
                    y_adj = _dampen_to_last_raw(last_raw, y_adj, damp=damp)

                outputs.append(y_adj.unsqueeze(1))  # [B,1]
                x_raw = _prepare_next_input(x_raw, y_adj,
                                            target_channel=self.target_channel,
                                            fill_mode=self.fill_mode)

        y_hat = torch.cat(outputs, dim=1)  # [B,H] RAW

        if DEBUG_FCAST:
            print(f"[FCAST-DBG] DONE: H={y_hat.size(1)}, var={_tvar(y_hat):.6g}, first5={_tfirst5(y_hat)}")

        if was_training:
            self.model.train()
        return y_hat
