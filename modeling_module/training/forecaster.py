# forecaster.py

import torch
from typing import Optional, Callable

from torch.utils.data import DataLoader  # noqa: F401  # (인터페이스 호환용)

DEBUG_FCAST = True


# -------------------- Utilities --------------------
def _tvar(t: torch.Tensor) -> float:
    """
    시간축(=dim=1) 분산의 배치 평균을 간단 확인용으로 계산.
    기대 shape: [B, H], [B, L], [B, H, *]도 허용(앞의 [B,H]로 맞춤)
    """
    if t.dim() >= 2:
        t2 = t.reshape(t.size(0), t.size(1), -1).mean(-1)  # [B, H]
        return t2.var(dim=1).mean().item()
    return float('nan')


def _tfirst5(t: torch.Tensor) -> str:
    """
    첫 배치의 앞 5개 시점 값을 프린트용 문자열로 반환.
    기대 shape: [B, H] 또는 [B, H, C]
    """
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
    """
    단순 주기성(sin/cos) 외생변수 생성: (H, 2)
    """
    t = torch.arange(start_idx, start_idx + H, device=device, dtype=torch.float32)
    exo = torch.stack([torch.sin(2 * torch.pi * t / period),
                       torch.cos(2 * torch.pi * t / period)], dim=-1)  # (H, 2)
    return exo


def _prepare_next_input(
    x_raw: torch.Tensor,
    y_step_raw: torch.Tensor,
    *,
    target_channel: int = 0,
    fill_mode: str = 'copy_last',   # 원시 스케일에서 last 복사 or zeros
) -> torch.Tensor:
    """
    x_raw: [B, L, C]  (원시 스케일)
    y_step_raw: [B]   (원시 스케일 예측값)
    """
    assert x_raw.dim() == 3, f"x must be [B, L, C], got {x_raw.shape}"
    B, L, C = x_raw.shape
    y_step_raw = y_step_raw.reshape(B, 1, 1)  # -> [B,1,1]

    if C == 1:
        new_token = y_step_raw
    else:
        last = x_raw[:, -1:, :].clone()
        if fill_mode == 'zeros':
            new_token = torch.zeros_like(last)
        else:
            new_token = last  # copy_last
        new_token[:, 0, target_channel] = y_step_raw[:, 0, 0]

    x_next = torch.cat([x_raw[:, 1:, :], new_token], dim=1)
    return x_next


def _winsorize_clamp(
    hist_n: torch.Tensor,     # [B, L]  (정규화 공간의 타깃 히스토리)
    y_step_n: torch.Tensor,   # [B]
    *,
    nonneg: bool = True,
    clip_q: tuple[float, float] = (0.05, 0.95),
    clip_mul: float = 4.0,
    max_growth: float = 0.05
) -> torch.Tensor:
    """
    히스토리 분위수 기반 + 직전값 대비 성장률 기반으로 y_step_n을 클램프.
    내부/입출력 모두 정규화 공간 기준.
    """
    hist = hist_n.float()
    y = y_step_n.float()

    B, L = hist.shape
    last = hist[:, -1]

    # 분위수 계산 안정화
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


def _dampen_to_last(
    last_n: torch.Tensor,     # [B] (정규화 공간)
    y_step_n: torch.Tensor,
    *,
    damp: float = 0.1
) -> torch.Tensor:
    """
    y_step_n과 직전값 last_n을 혼합해 급변을 완화. (정규화 공간)
    """
    if damp <= 0.0:
        return y_step_n
    return (1.0 - damp) * last_n + damp * y_step_n


def _guard_multiplicative(
    last_n: torch.Tensor,     # [B] (정규화 공간)
    y_raw_n: torch.Tensor,    # [B] (정규화 공간)
    *,
    max_step_up: float = 0.10,
    max_step_down: float = 0.20
) -> torch.Tensor:
    """
    로그-비율 도메인에서 상승/하락 비율 제한. (정규화 공간)
    last==0 케이스 보호를 위해 eps 사용.
    """
    eps = 1e-6
    last_safe = torch.clamp(last_n, min=eps)
    y_safe = torch.clamp(y_raw_n, min=eps)

    ratio = y_safe / last_safe
    log_ratio = torch.log(ratio)

    log_min = torch.log(torch.tensor(1.0 - max_step_down, device=last_n.device))
    log_max = torch.log(torch.tensor(1.0 + max_step_up, device=last_n.device))

    log_ratio = torch.clamp(log_ratio, min=log_min, max=log_max)
    y_guard = last_safe * torch.exp(log_ratio)
    return y_guard


# -------------------- Forecaster --------------------
class DMSForecaster:
    """
    DMS(Direct Multi-Step) + 필요 시 IMS(Iterated Multi-Step) 확장 예측기.

    설계 원칙 (RevIN을 모델 내부에서 'norm'만 수행, denorm은 Forecaster에서 수행)
    - 모델 입력/슬라이딩 윈도우는 '원시 스케일(raw)'로 유지한다.
    - 모델 출력은 '정규화 공간'이므로, 가드·윈저·댐핑 등은 정규화 공간에서 수행한다.
    - 슬라이딩 시에는 다음 입력을 위해 스텝별로 y_step_n → y_step_raw 로 denorm 하여 x_raw에 붙인다.
    - 최종 반환 y_hat은 원시 스케일(raw).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        target_channel: int = 0,
        fill_mode: str = "copy_last",  # {'copy_last', 'zeros'}  (raw space)
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
        self.global_t0 = 0  # 절대 인덱스(외생변수 기준 시작점)

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
                    y_full = q[..., 0]  # 첫 채널
            else:
                k = next(iter(y_full))
                y_full = y_full[k]
        return y_full

    def _normalize_by_horizon(self, y_full, B: int, H_hint: Optional[int] = None) -> torch.Tensor:
        """
        다양한 출력 텐서를 [B,H](정규화 공간)로 정규화.
        """
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
                    ch = min(self.target_channel, d1 - 1)
                    return y_full[:, ch, :]
                if d1 == H_hint and d2 == H_hint:
                    return y_full[:, :, 0]

            # H_hint 불명 → 보수 처리
            if d2 in (1, 3):
                return y_full[:, :, 1] if d2 == 3 else y_full[:, :, 0]
            ch = min(self.target_channel, d1 - 1)
            return y_full[:, ch, :]

        return y_full.reshape(B, -1)

    @torch.no_grad()
    def _context_features(self, x_raw: torch.Tensor) -> torch.Tensor:
        # 필요 시 encoder.input_proj로 컨텍스트 임베딩
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "input_proj"):
            return self.model.encoder.input_proj(x_raw)
        return x_raw

    def _get_h_hint(self) -> int:
        return int(getattr(self.model, "horizon", getattr(self.model, "output_horizon", 0)) or 0)

    def _call_model(self, x_raw: torch.Tensor, B: int) -> torch.Tensor:
        """
        모델은 내부에서 RevIN.norm을 수행한다고 가정. 입력은 'raw'를 그대로 넣는다.
        반환은 정규화 공간 [B,H].
        """
        H_hint = self._get_h_hint()
        if self.predict_fn is not None:
            y = self.predict_fn(x_raw)
            return self._normalize_by_horizon(y, B, H_hint)

        try:
            y = self.model(x_raw)
            return self._normalize_by_horizon(y, B, H_hint)
        except TypeError:
            try:
                y = self.model(x_raw, mode=(self.lmm_mode or "eval"))
                return self._normalize_by_horizon(y, B, H_hint)
            except TypeError:
                try:
                    y = self.model(x_raw, future_exo=None)
                    return self._normalize_by_horizon(y, B, H_hint)
                except TypeError:
                    y = self.model(x_raw, future_exo=None, mode=(self.lmm_mode or "eval"))
                    return self._normalize_by_horizon(y, B, H_hint)

    def _try_model_call(self, x_raw: torch.Tensor, future_exo: torch.Tensor, B: int) -> torch.Tensor:
        try:
            return self.model(x_raw, future_exo=future_exo, mode=(self.lmm_mode or "eval"))
        except TypeError:
            pass
        try:
            return self.model(x_raw, future_exo=future_exo)
        except TypeError:
            pass
        try:
            return self.model(x_raw, mode=(self.lmm_mode or "eval"))
        except TypeError:
            pass
        return self.model(x_raw)

    def _call_with_exo(self, x_raw: torch.Tensor, B: int, H_need: int, step_offset: int) -> torch.Tensor:
        """
        future_exo_cb가 있으면 (B,H,exo)로 주입하여 모델 호출.
        반환은 '정규화 공간' [B,H].
        """
        H_hint = H_need or self._get_h_hint()
        if self.future_exo_cb is None:
            return self._call_model(x_raw, B)
        t0 = self.global_t0 + step_offset
        exo = self.future_exo_cb(t0, H_hint).to(x_raw.device)  # (H, exo_dim)
        exo = exo.unsqueeze(0).expand(B, -1, -1)
        y_full = self._try_model_call(x_raw, exo, B)
        return self._normalize_by_horizon(y_full, B, H_hint)

    def _denorm_like_revin(self, y_any: torch.Tensor | dict) -> torch.Tensor | dict:
        """
        model.revin_layer(…, 'denorm')을 사용해 정규화 공간→raw로 변환.
        정규화 통계는 '바로 직전 norm 호출(=마지막 model(...), 혹은 이 함수 내 임시 norm)'의 것을 사용.

        - 1D([B]) / 2D([B,H]) / 3D([B,H,k]) 안전 지원
        - dict({"point":[B,H], "q":[B,H,3]}) 형태도 지원
        """
        def _denorm_tensor(t: torch.Tensor) -> torch.Tensor:
            if not hasattr(self.model, "revin_layer"):
                return t  # RevIN 미보유 모델: 원본 반환

            # 표준화: RevIN은 [B,*,C] 형식을 기대하므로 1D/2D도 [B,H,1]로 승격
            if t.dim() == 1:          # [B]
                t1 = t.view(t.size(0), 1, 1)
                out = self.model.revin_layer(t1, 'denorm')  # [B,1,1]
                return out.view(t.size(0))
            if t.dim() == 2:          # [B,H]
                t1 = t.unsqueeze(-1)  # [B,H,1]
                out = self.model.revin_layer(t1, 'denorm')  # [B,H,1]
                return out.squeeze(-1)
            if t.dim() == 3 and t.size(-1) in (1, 3):  # [B,H,1] or [B,H,3]
                if t.size(-1) == 1:
                    out = self.model.revin_layer(t, 'denorm')  # [B,H,1]
                    return out.squeeze(-1)
                outs = []
                for i in range(t.size(-1)):
                    ti = t[..., i].unsqueeze(-1)              # [B,H,1]
                    oi = self.model.revin_layer(ti, 'denorm') # [B,H,1]
                    outs.append(oi.squeeze(-1))
                return torch.stack(outs, dim=-1)              # [B,H,3]
            # 기타 형상은 그대로 시도
            return self.model.revin_layer(t, 'denorm')

        if isinstance(y_any, dict):
            y_any = dict(y_any)  # shallow copy
            if "point" in y_any:
                y_any["point"] = _denorm_tensor(y_any["point"])
            if "q" in y_any:
                y_any["q"] = _denorm_tensor(y_any["q"])
            return y_any
        return _denorm_tensor(y_any)

    # ---------- public: forecasting ----------
    @torch.no_grad()
    def forecast_DMS_to_IMS(
        self,
        x_init: Optional[torch.Tensor] = None,   # 호환용(기존 유틸이 x_init=로 호출)
        *,
        x: Optional[torch.Tensor] = None,        # 호환용(다른 호출부에서 x=로 넘길 수 있음)
        horizon: Optional[int] = None,
        device: Optional[torch.device] = None,
        extend: str = "ims",                    # {'ims','error'}
        context_policy: str = "once",           # {'once','per_step','off'}
        y_true: Optional[torch.Tensor] = None,  # (정규화 공간 기준) Teacher Forcing 타깃
        teacher_forcing_ratio: float = 0.0,

        # 안정화 토글 & 파라미터 (정규화 공간에서 적용)
        use_winsor: bool = False,
        use_multi_guard: bool = False,
        use_dampen: bool = False,
        winsor_q: tuple = (0.05, 0.95),
        winsor_mul: float = 4.0,
        winsor_growth: float = 3.0,
        max_step_up: float = 0.10,
        max_step_down: float = 0.40,
        damp: float = 0.5,
    ) -> torch.Tensor:
        """
        DMS 한 번으로 Hm 구간을 만들고, H>Hm이면 IMS로 초과 구간을 생성한다.

        입력:
          - x_init 또는 x 중 하나를 필수로 전달 (둘 다 주면 x_init 우선)
          - 입력 텐서는 '원시 스케일(raw)'

        핵심:
          - 모델은 내부 RevIN.norm을 수행 → 입력은 항상 raw.
          - 예측 y는 정규화 공간 → 가드/윈저/댐핑은 정규화 공간에서 수행.
          - 슬라이딩을 위해 스텝별로 y_step_n을 denorm하여 x_raw에 붙인다.
          - 최종 반환 y_hat은 raw.
        """
        # ---- 입력 통합 ----
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

        # TTM context
        if (self.ttm is not None) and (context_policy in ("once", "per_step")):
            if context_policy == "once":
                self.ttm.add_context(self._context_features(x_raw))

        # 모델 출력 길이 Hm 추정 (exo 없이 → 실패 시 exo 포함)
        def _probe_hm_safe() -> int:
            try:
                return self._call_model(x_raw, B).size(1)
            except Exception:
                H_guess = self._get_h_hint() or 120
                return self._call_with_exo(x_raw, B, H_guess, step_offset=0).size(1)

        Hm = _probe_hm_safe()
        H = int(horizon) if horizon is not None else Hm

        # DMS 본블록(정규화 공간)
        y_block_n = self._call_with_exo(x_raw, B, Hm, step_offset=0)  # [B,Hm] (normalized)

        if DEBUG_FCAST:
            print(f"[FCAST-DBG] DMS block: Hm={y_block_n.size(1)}, "
                  f"var(Hm)={_tvar(y_block_n):.6g}, first5={_tfirst5(y_block_n)}")

        outputs_raw = []
        use_tf = (y_true is not None) and (teacher_forcing_ratio > 0.0)
        if use_tf:
            y_true = y_true.to(device).float()  # 정규화 공간 기준이라고 가정

        # 본구간: min(Hm, H) 스텝 공동 루프
        use_len = min(Hm, H)
        for t in range(use_len):
            # per-step TTM
            if (self.ttm is not None) and (context_policy == "per_step"):
                self.ttm.add_context(self._context_features(x_raw))

            # 정규화 공간에서의 원시 예측
            y_raw_n = y_block_n[:, t]  # [B] normalized
            if DEBUG_FCAST and t < 5:
                print(f"[FCAST-DBG] DMS step={t}: raw_n={float(y_raw_n[0]):.6g}")

            # 정규화 공간에서의 안정화/TF
            if use_tf and (t < (y_true.shape[1] if y_true.dim() > 1 else 0)) \
               and (torch.rand(1).item() < teacher_forcing_ratio):
                y_step_n = y_true[:, t]  # teacher forcing (normalized)
            else:
                # 정규화 히스토리를 만들기 위해 현재 raw x를 한시적으로 norm
                # (모델 내부에서도 norm하지만, 여기서는 히스토리 통계용)
                if hasattr(self.model, "revin_layer"):
                    x_n_tmp = self.model.revin_layer(x_raw, "norm")  # 통계 업데이트(현 시점)
                    hist_n = x_n_tmp[:, :, self.target_channel]
                else:
                    # RevIN이 없으면 raw에서 직접 사용(덜 안정적일 수 있음)
                    hist_n = x_raw[:, :, self.target_channel]
                last_n = hist_n[:, -1]

                y_step_n = y_raw_n
                if use_winsor:
                    y_step_n = _winsorize_clamp(
                        hist_n, y_step_n,
                        nonneg=True, clip_q=winsor_q,
                        clip_mul=winsor_mul, max_growth=winsor_growth
                    )
                if use_multi_guard:
                    y_step_n = _guard_multiplicative(
                        last_n, y_step_n,
                        max_step_up=max_step_up, max_step_down=max_step_down
                    )
                if use_dampen:
                    y_step_n = _dampen_to_last(last_n, y_step_n, damp=damp)

            # 다음 입력을 위해 raw로 변환하여 슬라이딩
            y_step_raw = self._denorm_like_revin(y_step_n)  # [B] raw
            if isinstance(y_step_raw, dict):
                # 이 경로는 사실상 발생하지 않음(여기선 텐서만 옴). 안전장치.
                y_step_raw = y_step_raw.get("point", None)
                if y_step_raw is None:
                    raise RuntimeError("denorm result is not a tensor.")
            outputs_raw.append(y_step_raw.unsqueeze(1))
            x_raw = _prepare_next_input(
                x_raw, y_step_raw,
                target_channel=self.target_channel,
                fill_mode=self.fill_mode
            )

        # H > Hm: IMS 구간
        if H > Hm:
            if extend not in ("ims", "error"):
                raise ValueError("extend must be 'ims' or 'error'")
            if extend == "error":
                raise ValueError(f"horizon ({H}) > model_output ({Hm}). "
                                 f"Set extend='ims' to extend autoregressively.")

            remaining = H - use_len
            for t in range(remaining):
                if (self.ttm is not None) and (context_policy == "per_step"):
                    self.ttm.add_context(self._context_features(x_raw))

                # 이미 생성한 길이(use_len) + IMS 상대 step(t)만큼 step_offset 적용
                y_full_n = self._call_with_exo(x_raw, B, Hm, step_offset=(use_len + t))  # [B,Hm] normalized
                y_raw_n = y_full_n[:, 0]  # 다음 한 스텝만 사용

                if DEBUG_FCAST and t < 5:
                    print(f"[FCAST-DBG] IMS step={t}: raw_n={float(y_raw_n[0]):.6g}")

                # 안정화용 정규화 히스토리
                if hasattr(self.model, "revin_layer"):
                    x_n_tmp = self.model.revin_layer(x_raw, "norm")  # 통계 업데이트(현 시점)
                    hist_n = x_n_tmp[:, :, self.target_channel]
                else:
                    hist_n = x_raw[:, :, self.target_channel]
                last_n = hist_n[:, -1]

                y_step_n = y_raw_n
                if use_winsor:
                    y_step_n = _winsorize_clamp(
                        hist_n, y_step_n,
                        nonneg=True, clip_q=winsor_q,
                        clip_mul=winsor_mul, max_growth=winsor_growth
                    )
                if use_multi_guard:
                    y_step_n = _guard_multiplicative(
                        last_n, y_step_n,
                        max_step_up=max_step_up, max_step_down=max_step_down
                    )
                if use_dampen:
                    y_step_n = _dampen_to_last(last_n, y_step_n, damp=damp)

                # raw로 변환하여 슬라이딩
                y_step_raw = self._denorm_like_revin(y_step_n)
                if isinstance(y_step_raw, dict):
                    y_step_raw = y_step_raw.get("point", None)
                    if y_step_raw is None:
                        raise RuntimeError("denorm result is not a tensor.")
                outputs_raw.append(y_step_raw.unsqueeze(1))
                x_raw = _prepare_next_input(
                    x_raw, y_step_raw,
                    target_channel=self.target_channel,
                    fill_mode=self.fill_mode
                )

        # [B,H] raw 예측을 반환
        y_hat = torch.cat(outputs_raw, dim=1)  # [B, H] (raw)

        if DEBUG_FCAST:
            print(f"[FCAST-DBG] DONE: H={y_hat.size(1)}, var={_tvar(y_hat):.6g}, first5={_tfirst5(y_hat)}")

        if was_training:
            self.model.train()
        return y_hat


# -------------------- 사용 예시 (주석) --------------------
# model = YourModel(...)  # 내부에서 RevIN.norm 수행, denorm은 하지 않음
# forecaster = DMSForecaster(model, target_channel=0, fill_mode='zeros')
# x_init_raw = ...  # [B,L,C] (원시 스케일)
# y_hat = forecaster.forecast_DMS_to_IMS(
#     x_init=x_init_raw,       # 또는 x=x_init_raw
#     horizon=120,
#     extend='ims',
#     use_winsor=True,
#     use_multi_guard=True,
#     use_dampen=True,
#     winsor_q=(0.05, 0.95),
#     winsor_mul=4.0,
#     winsor_growth=2.0,
#     max_step_up=0.10,
#     max_step_down=0.60,
#     damp=0.4
# )
