import torch
from typing import Optional, Callable, Tuple

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

    # ---------- 공동 internal helpers ----------

    def _unwrap_output(self, y_full):
        if isinstance(y_full, dict):
            if "point" in y_full:
                y_full = y_full["point"]
            elif "q" in y_full:
                q = y_full["q"]
                if torch.is_tensor(q):
                    if q.dim() == 3 and q.size(-1) >= 3:
                        y_full = q[..., 1]  # q50
                    else:
                        y_full = q[..., 0]
                else:
                    # dict(q10,q50,q90)인 경우 q50 우선
                    y_full = q.get("q50", next(iter(q.values())))
            else:
                k = next(iter(y_full))
                y_full = y_full[k]
        return y_full

    def _get_h_hint(self) -> int:
        return int(getattr(self.model, "horizon",
                           getattr(self.model, "output_horizon", 0)) or 0)

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

    def _call_model_point(self, x_raw: torch.Tensor, B: int,
                           future_exo: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Point 예측용: dict/q까지 풀어서 [B,H] RAW 반환."""
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

    def _safe_call_any(self, x: torch.Tensor, future_exo: Optional[torch.Tensor] = None):
        """
        Quantile용: dict / 텐서 / tuple 모두 그대로 반환 (언랩 X).
        """
        tries = []
        if future_exo is not None:
            tries += [dict(future_exo=future_exo), {}]
        else:
            tries += [dict(), ]
        for args in tries:
            try:
                return self.model(x, **args)
            except TypeError:
                continue
        return self.model(x)

    def _infer_model_Hm_quantile(
        self,
        x_raw: torch.Tensor,
        device: torch.device,
        future_exo_cb,
        default: int = 120,
    ) -> int:
        """
        Quantile 출력 기준으로 DMS horizon(Hm) 추론.
        """
        for k in ("horizon", "output_horizon", "H", "Hm"):
            if hasattr(self.model, k):
                try:
                    v = int(getattr(self.model, k))
                    if v > 0:
                        return v
                except Exception:
                    pass

        B, L, C = x_raw.shape
        x_dev = x_raw.to(device)
        out = None
        try:
            out = self.model(x_dev)
        except TypeError:
            if future_exo_cb is not None:
                ex = future_exo_cb(0, default, device=device)
                if torch.is_tensor(ex):
                    ex = ex.to(device)
                if ex.dim() == 2:
                    ex = ex.unsqueeze(0).expand(B, -1, -1)
                out = self.model(x_dev, future_exo=ex)

        if out is None:
            return default

        # 첫 usable만 보고 shape 추론
        if isinstance(out, (tuple, list)):
            for t in out:
                if torch.is_tensor(t) or isinstance(t, dict):
                    out = t
                    break

        if isinstance(out, dict):
            if "q" in out:
                out = out["q"]
            elif "point" in out:
                out = out["point"]

        if not torch.is_tensor(out):
            return default
        if out.dim() == 2:
            return out.shape[1]
        if out.dim() == 3:
            _, a, b = out.shape
            if a in (3, 5, 9):
                return b
            if b in (3, 5, 9):
                return a
            return b
        return default

    # ---------- public: point DMS→IMS ----------

    @torch.no_grad()
    def forecast_DMS_to_IMS(
        self,
        x_init: Optional[torch.Tensor] = None,   # preferred name
        *,
        x: Optional[torch.Tensor] = None,        # alias
        horizon: Optional[int] = None,
        device: Optional[torch.device] = None,
        extend: str = "ims",                    # {'ims','error'}
        context_policy: str = "per_step",       # {'once','per_step','off'}
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
                self.ttm.add_context(x_raw)

        # --- Hm estimation ---
        def _probe_hm_safe() -> int:
            try:
                return self._call_model_point(x_raw, B).size(1)
            except Exception:
                H_guess = self._get_h_hint() or 120
                exo = None
                if self.future_exo_cb is not None:
                    exo = self.future_exo_cb(self.global_t0, H_guess).to(x_raw.device).unsqueeze(0).expand(B, -1, -1)
                return self._call_model_point(x_raw, B, exo).size(1)

        Hm = _probe_hm_safe()
        H = int(horizon) if horizon is not None else Hm

        # --- DMS block (RAW) ---
        def _call_with_exo(xr: torch.Tensor, need: int, step_offset: int):
            exo = None
            if self.future_exo_cb is not None:
                t0 = self.global_t0 + step_offset
                exo = self.future_exo_cb(t0, need).to(xr.device)  # (H, exo)
                exo = exo.unsqueeze(0).expand(B, -1, -1)
            return self._call_model_point(xr, B, future_exo=exo)  # [B,need] RAW

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

            x_raw = _prepare_next_input(
                x_raw, y_adj,
                target_channel=self.target_channel,
                fill_mode=self.fill_mode
            )

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

                y_full_raw = _call_with_exo(x_raw, Hm, step_offset=(use_len + t))  # [B,Hm] RAW
                y_step_raw = y_full_raw[:, 0]

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
                x_raw = _prepare_next_input(
                    x_raw, y_adj,
                    target_channel=self.target_channel,
                    fill_mode=self.fill_mode
                )

        y_hat = torch.cat(outputs, dim=1)  # [B,H] RAW

        if DEBUG_FCAST:
            print(f"[FCAST-DBG] DONE: H={y_hat.size(1)}, var={_tvar(y_hat):.6g}, first5={_tfirst5(y_hat)}")

        if was_training:
            self.model.train()
        return y_hat

    # ---------- public: quantile DMS→IMS ----------

    @torch.no_grad()
    def quantile_DMS_to_IMS(
        self,
        x_init: torch.Tensor,          # [B,L] or [B,L,1] or [B,L,C]
        *,
        horizon: int,                  # 최종 원하는 예측 길이 (예: 120)
        device: str | torch.device = None,
        future_exo_cb=None,           # exo_cb(t0: int, H: int, device) -> Tensor[H, E]
        target_channel: Optional[int] = None,
        fill_mode: Optional[str] = None,  # {'copy_last','zeros'}
        feed_quantile: str = "q50",    # {'q10','q50'}

        # --- 하강 유도 / 가드 관련 하이퍼들 ---
        use_winsor: bool = True,
        use_multi_guard: bool = True,
        use_dampen: bool = True,
        winsor_q: Tuple[float, float] = (0.10, 0.85),
        winsor_mul: float = 2.0,
        max_step_up: float = 0.02,
        max_step_down: float = 0.30,
        damp: float = 0.6,
        down_bias: float = 0.005,
        decay_rate: float = 0.0,
        eps: float = 1e-6,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantile 모델에 대해:
          1) 모델이 학습한 DMS horizon(Hm)을 자동으로 추론
          2) 앞 min(H, Hm) 구간은 DMS 블록을 그대로 사용
          3) 남은 H - min(H,Hm) 구간은 IMS(1-step autoregressive)로 연장
          4) 슬라이딩 윈도우에 넣는 값은 성장률 기반 guard (winsor + multiguard + dampen) 적용

        반환: (q10, q50, q90) 각 [B, H]
        """
        device = device or next(self.model.parameters()).device
        device = torch.device(device)
        cb = future_exo_cb or self.future_exo_cb
        tgt_ch = self.target_channel if target_channel is None else target_channel
        fm = self.fill_mode if fill_mode is None else fill_mode

        # -------- small helpers (quantile 전용) --------
        def _first_usable(x):
            if isinstance(x, (tuple, list)):
                for t in x:
                    if torch.is_tensor(t) or isinstance(t, dict):
                        return t
            return x

        def _to_q3d(out_any: object) -> torch.Tensor:
            out_any = _first_usable(out_any)
            # Tensor
            if torch.is_tensor(out_any):
                if out_any.dim() != 3:
                    raise RuntimeError(f"Expect 3D tensor, got {tuple(out_any.shape)}")
                B_, A, B2 = out_any.shape
                if A in (3, 5, 9) or B2 in (3, 5, 9):
                    return out_any
                raise RuntimeError(f"3D tensor지만 quantile 축(3/5/9)을 찾을 수 없음: {tuple(out_any.shape)}")

            # dict
            if isinstance(out_any, dict):
                if "q" in out_any:
                    q = out_any["q"]
                    if torch.is_tensor(q):
                        if q.dim() != 3:
                            raise RuntimeError(f"dict['q'] must be 3D, got {tuple(q.shape)}")
                        return q
                    if isinstance(q, dict):
                        q10 = q.get("q10")
                        q50 = q.get("q50")
                        q90 = q.get("q90")
                        if not all(torch.is_tensor(t) for t in (q10, q50, q90)):
                            raise RuntimeError("dict['q'] dict 형식인데 q10/q50/q90 텐서가 없습니다.")
                        def _S(t: torch.Tensor):
                            if t.dim() == 1:
                                return t.unsqueeze(0)
                            if t.dim() == 2:
                                return t
                            if t.dim() == 3 and t.size(-1) == 1:
                                return t.squeeze(-1)
                            raise RuntimeError(f"Unsupported q tensor shape {tuple(t.shape)}")
                        q10_2d = _S(q10)
                        q50_2d = _S(q50)
                        q90_2d = _S(q90)
                        return torch.stack([q10_2d, q50_2d, q90_2d], dim=1)  # (B,3,H)
                raise RuntimeError("dict 출력이지만 'q' 키를 찾지 못했습니다.")

            raise RuntimeError(f"Cannot convert output type={type(out_any)} to quantile tensor.")

        def _extract_block_q(out_any: object) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q3d = _to_q3d(out_any)   # (B,Q,Hm) or (B,Hm,Q)
            if q3d.shape[1] in (3, 5, 9):      # (B,Q,Hm)
                B_, Qn, Hm_ = q3d.shape
                if Qn == 3:
                    i10, i50, i90 = 0, 1, 2
                elif Qn == 5:
                    i10, i50, i90 = 1, 2, 3
                else:
                    i10, i50, i90 = 1, 4, 7
                q10 = q3d[:, i10, :]
                q50 = q3d[:, i50, :]
                q90 = q3d[:, i90, :]
            elif q3d.shape[2] in (3, 5, 9):    # (B,Hm,Q)
                B_, Hm_, Qn = q3d.shape
                if Qn == 3:
                    i10, i50, i90 = 0, 1, 2
                elif Qn == 5:
                    i10, i50, i90 = 1, 2, 3
                else:
                    i10, i50, i90 = 1, 4, 7
                q10 = q3d[:, :, i10]
                q50 = q3d[:, :, i50]
                q90 = q3d[:, :, i90]
            else:
                raise RuntimeError(f"Cannot infer quantile axis from shape {tuple(q3d.shape)}")
            return q10, q50, q90

        def _apply_growth_guard(x_win: torch.Tensor, y_step: torch.Tensor) -> torch.Tensor:
            """
            슬라이딩 윈도우용 next-level 값을 만드는 guard.
            """
            hist = x_win[:, :, tgt_ch]   # [B,L]
            last = hist[:, -1]

            last_safe = last.clamp_min(eps)
            base_growth = (y_step / last_safe) - 1.0  # 상대 변화율

            g = base_growth

            # 1) winsor
            if use_winsor and g.numel() >= 4:
                q_lo_t, q_hi_t = torch.quantile(
                    g,
                    torch.tensor([winsor_q[0], winsor_q[1]], device=g.device)
                )
                iqr = (q_hi_t - q_lo_t).clamp_min(1e-6)
                lo = q_lo_t - winsor_mul * iqr
                hi = q_hi_t + winsor_mul * iqr
                g = torch.clamp(g, lo, hi)

            # 2) bias + decay + multi_guard
            g = g - down_bias
            if decay_rate > 0.0:
                g = g - float(decay_rate)

            if use_multi_guard:
                g = torch.clamp(g, -max_step_down, max_step_up)

            # 3) damp
            if use_dampen and damp > 0.0:
                g = damp * g

            y_next = last_safe * (1.0 + g)
            return y_next

        # ----------------- 준비: x, H, Hm -----------------
        self.model.to(device).eval()
        x_raw = x_init.to(device).float().clone()
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(-1)
        B, L, C = x_raw.shape

        Hm = self._infer_model_Hm_quantile(x_raw, device, cb, default=horizon or 120)
        H = int(horizon)
        use_len = min(H, Hm)

        if debug:
            print(f"[Q-DMS/IMS] inferred Hm={Hm}, requested H={H}, "
                  f"use_len(DMS)={use_len}, IMS_len={max(0, H-use_len)}")

        def _call_block(x_win: torch.Tensor, step_offset: int):
            exo = None
            if cb is not None:
                ex = cb(step_offset, Hm, device=device)  # (Hm, E)
                if torch.is_tensor(ex):
                    ex = ex.to(device)
                if ex.dim() == 2:
                    exo = ex.unsqueeze(0).expand(x_win.size(0), -1, -1)
            out = self._safe_call_any(x_win.to(device), future_exo=exo)
            return _extract_block_q(out)  # [B,Hm] x3

        # 초기 DMS 블록
        q10_block, q50_block, q90_block = _call_block(x_raw, step_offset=0)

        q10_seq, q50_seq, q90_seq = [], [], []

        # -------- DMS phase --------
        for t in range(use_len):
            q10_t = q10_block[:, t]
            q50_t = q50_block[:, t]
            q90_t = q90_block[:, t]

            if feed_quantile == "q10":
                y_step = q10_t
            else:
                y_step = q50_t

            y_next = _apply_growth_guard(x_raw, y_step)

            q10_seq.append(q10_t.unsqueeze(1))
            q50_seq.append(q50_t.unsqueeze(1))
            q90_seq.append(q90_t.unsqueeze(1))

            x_raw = _prepare_next_input(
                x_raw, y_next,
                target_channel=tgt_ch,
                fill_mode=fm
            )

            if debug and t < 5:
                print(
                    f"[Q-DMS] t={t} "
                    f"q10={float(q10_t[0]):.6g} q50={float(q50_t[0]):.6g} q90={float(q90_t[0]):.6g} "
                    f"y_next={float(y_next[0]):.6g}"
                )

        # -------- IMS extension --------
        if H > use_len:
            remain = H - use_len
            for k in range(remain):
                step_offset = use_len + k
                qb10, qb50, qb90 = _call_block(x_raw, step_offset=step_offset)
                q10_t = qb10[:, 0]
                q50_t = qb50[:, 0]
                q90_t = qb90[:, 0]

                if feed_quantile == "q10":
                    y_step = q10_t
                else:
                    y_step = q50_t

                y_next = _apply_growth_guard(x_raw, y_step)

                q10_seq.append(q10_t.unsqueeze(1))
                q50_seq.append(q50_t.unsqueeze(1))
                q90_seq.append(q90_t.unsqueeze(1))

                x_raw = _prepare_next_input(
                    x_raw, y_next,
                    target_channel=tgt_ch,
                    fill_mode=fm
                )

                if debug and k < 5:
                    print(
                        f"[Q-IMS] k={k} (global t={use_len+k}) "
                        f"q10={float(q10_t[0]):.6g} q50={float(q50_t[0]):.6g} q90={float(q90_t[0]):.6g} "
                        f"y_next={float(y_next[0]):.6g}"
                    )

        q10 = torch.cat(q10_seq, dim=1)
        q50 = torch.cat(q50_seq, dim=1)
        q90 = torch.cat(q90_seq, dim=1)

        return q10, q50, q90