import torch
from typing import Optional, Callable

from torch.utils.data import DataLoader

def make_calendar_exo(start_idx: int, H: int, period: int = 12, device = 'cpu'):
    t = torch.arange(start_idx, start_idx + H, device = device, dtype = torch.float32)
    # 월 주기 sin/cos
    exo = torch.stack([torch.sin(2*torch.pi*t/period), torch.cos(2*torch.pi*t/period)], dim = -1) # (H, 2)
    return exo # (H, 2)

def _prepare_next_input(
        x: torch.Tensor,
        y_step: torch.Tensor,
        target_channel: int = 0,
        fill_mode: str = 'copy_last',
) -> torch.Tensor:
    '''
    현재 윈도우 x에 대해 한 스텝 예측값 y_step을 마지막 시점으로 붙여 넣고,
    가장 오래된 시점을 버려 윈도우를 한 칸 전진.
    :param x: [B, L, C] 입력 윈도우 (Batch, Lookback, Channel)
    :param y_step: 이번 스텝에 예측된 target value.
    :param target_channel: int
        target 변수가 위치한 채널 인덱스 (기본 0)
    :param fill_mode: {'copy_last', 'zeros'}
        다변량 (C>1)일 때, 타깃 외 채널을 새 시점에 채우는 방식
        - 'copy_last': 직전 시점 값을 복사
        - 'zeros': 0으로 채움
    :return:
    x_next: [B, L, C] 한 칸 전진된 다음 입력 윈도우
    '''
    assert x.dim() == 3, f"x must be [B, L, C], got {x.shape}"
    B, L, C = x.shape
    y_step = y_step.reshape(B, 1, 1)  # -> [B, 1, 1]

    if C == 1:
        new_token = y_step  # 단변량이면 예측값만 넣으면 됨

    else:
        last = x[:, -1:, :].clone()  # 직전 시점 값
        if fill_mode == 'zeros':
            new_token = torch.zeros_like(last)
        else:
            new_token = last  # copy_last

        new_token[:, 0, target_channel] = y_step[:, 0, 0]

    # Window slide + new series
    x_next = torch.cat([x[:, 1:, :], new_token], dim=1)
    return x_next


def _winsorize_clamp(
        hist: torch.Tensor,  # [B, L] (target 채널 히스토리)
        y_step: torch.Tensor,  # [B]
        nonneg: bool = True,
        clip_q: tuple[float, float] = (0.05, 0.95),
        clip_mul: float = 4.0,  # 상위 분위수 배수 상한
        max_growth: float = 0.05  # 직전값 대비 최대 성장배율 max_growth=2.0 ~ 2.5
) -> torch.Tensor:
    """
    히스토리 분위수/직전값 대비 성장률로 y_step을 winsorize + clamp.
    (nan_to_num 대신 torch.where로 텐서별 대체)
    """
    # 안전하게 float로
    hist = hist.float()
    y = y_step.float()

    B, L = hist.shape
    last = hist[:, -1]  # [B]

    # 히스토리 내 비유한/비유효값을 직전값으로 대체하여 분위수 계산 안정화
    hist_safe = torch.where(torch.isfinite(hist), hist, last.unsqueeze(1))

    q_lo = torch.quantile(hist_safe, clip_q[0], dim=1)  # [B]
    q_hi = torch.quantile(hist_safe, clip_q[1], dim=1)  # [B]

    min_cap = torch.zeros_like(q_lo) if nonneg else q_lo
    cap_quant = q_hi * clip_mul
    cap_growth = torch.where(last > 0, last * max_growth, cap_quant)
    max_cap = torch.minimum(cap_quant, cap_growth)  # [B]

    # 요소별로 nan/±inf를 안전 값으로 치환
    y = torch.where(torch.isnan(y), last, y)
    y = torch.where(torch.isposinf(y), max_cap, y)
    y = torch.where(torch.isneginf(y), min_cap, y)

    # 최종 클램프
    y = torch.clamp(y, min=min_cap, max=max_cap)
    return y


def _dampen_to_last(
        last: torch.Tensor,  # [B]
        y_step: torch.Tensor,
        damp: float = 0.1  # 0(무효)~1(그대로), 0.3~0.7 권장
) -> torch.Tensor:
    """
    예측값을 직전 관측과 혼합하여 급변을 완화.
    """
    if damp <= 0.0:
        return y_step
    return (1.0 - damp) * last + damp * y_step


def _guard_multiplicative(
        last: torch.Tensor,  # [B]
        y_raw: torch.Tensor,  # [B]
        max_step_up: float = 0.10,  # 스텝당 최대 상승비율 (예: +10%)
        max_step_down: float = 0.20  # 스텝당 최대 하락비율 (예: -20%)
) -> torch.Tensor:
    """
    직전값 대비 예측 비율을 log 도메인에서 클램프.
    last=0 보호를 위해 eps 사용.
    """
    eps = 1e-6
    last_safe = torch.clamp(last, min=eps)
    y_safe = torch.clamp(y_raw, min=eps)  # 비율/로그 계산 안정화

    ratio = y_safe / last_safe  # [B]
    log_ratio = torch.log(ratio)  # [B]

    min_ratio = 1.0 - max_step_down  # 예: 0.8
    max_ratio = 1.0 + max_step_up  # 예: 1.1
    log_min = torch.log(torch.tensor(min_ratio, device=last.device))
    log_max = torch.log(torch.tensor(max_ratio, device=last.device))

    log_ratio = torch.clamp(log_ratio, min=log_min, max=log_max)
    y_guard = last_safe * torch.exp(log_ratio)
    return y_guard


class DMSForecaster:
    """
    Titan/LMM 계열을 위한 DMS(Direct Multi-Step) 예측기 (+미래 exo/TTM/안정화 토글).
    - model(x) 1회로 [B, Hm] 출력 사용
    - horizon > Hm 이면 extend='ims'로 초과구간을 IMS로 이어붙임
    - future_exo_cb(start_idx, Hm) -> (Hm, exo_dim) 콜백 지원
    - global_t0: 예측 시작의 '절대 인덱스' (필요시 호출 전 세팅)
    """

    def _probe_hm(self, x: torch.Tensor, B: int) -> int:
        """
        모델의 실제 출력 길이(Hm)를 안전하게 프로빙.
        - model(x)
        - model(x, mode=...)
        - model(x, future_exo=None)
        - model(x, future_exo=None, mode=...)
        순으로 시도해서 첫 성공 출력의 size(1)를 반환.
        """
        # 1) model(x)
        try:
            y = self.model(x)
            return self._normalize_out(y, B).size(1)
        except TypeError:
            pass
        # 2) model(x, mode=...)
        try:
            y = self.model(x, mode=(self.lmm_mode or "eval"))
            return self._normalize_out(y, B).size(1)
        except TypeError:
            pass
        # 3) model(x, future_exo=None)
        try:
            y = self.model(x, future_exo=None)
            return self._normalize_out(y, B).size(1)
        except TypeError:
            pass
        # 4) model(x, future_exo=None, mode=...)
        y = self.model(x, future_exo=None, mode=(self.lmm_mode or "eval"))
        return self._normalize_out(y, B).size(1)

    def __init__(self,
                 model: torch.nn.Module,
                 target_channel: int = 0,
                 fill_mode: str = "copy_last",
                 lmm_mode: Optional[str] = None,
                 predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 ttm: Optional[object] = None,
                 future_exo_cb: Optional[Callable[[int, int], torch.Tensor]] = lambda s, h: make_calendar_exo(s, h, period=12)):
        self.model = model
        self.target_channel = target_channel
        self.fill_mode = fill_mode
        self.lmm_mode = lmm_mode
        self.predict_fn = predict_fn
        self.ttm = ttm
        self.future_exo_cb = future_exo_cb
        self.global_t0 = 0  # 외생변수 기준 시작 인덱스

    @torch.no_grad()
    def _context_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "input_proj"):
            return self.model.encoder.input_proj(x)
        return x

    def warmup(self,
               x_warm: torch.Tensor,
               y_warm: Optional[torch.Tensor] = None,
               steps: int = 1,
               add_context: bool = True,
               adapt: bool = True) -> None:
        if self.ttm is None:
            return
        if add_context:
            x_ctx = self._context_features(x_warm)
            self.ttm.add_context(x_ctx)
        if adapt and (y_warm is not None):
            _ = self.ttm.adapt(x_warm, y_warm, steps=steps)

    # ---------- 공통 유틸 ----------

    def _normalize_out(self, y_full: torch.Tensor, B: int) -> torch.Tensor:
        # 최종 형태는 [B, H]
        if y_full.dim() == 1:
            return y_full.view(B, -1)
        if y_full.dim() == 2:
            return y_full  # [B, H]

        if y_full.dim() == 3:
            # 두 축 길이
            d1, d2 = y_full.size(1), y_full.size(2)

            # 1) (B, 1, H)
            if d1 == 1:
                return y_full[:, 0, :]  # -> [B, H]
            # 2) (B, H, 1)
            if d2 == 1:
                return y_full[:, :, 0]  # -> [B, H]

            # 3) (B, C, H) vs (B, H, C) 추정:
            #    일반적으로 '시간축'이 더 김. 더 긴 쪽을 H로 간주
            if d2 >= d1:  # (B, H, C)로 가정
                ch = min(self.target_channel, d2 - 1)  # 안전 인덱스
                return y_full[:, :, ch]  # -> [B, H]
            else:  # (B, C, H)로 가정
                ch = min(self.target_channel, d1 - 1)
                return y_full[:, ch, :]  # -> [B, H]

        # fallback
        return y_full.reshape(B, -1)

    def _call_model(self, x: torch.Tensor, B: int) -> torch.Tensor:
        if self.predict_fn is not None:
            y_full = self.predict_fn(x)
        else:
            try:
                y_full = self.model(x)
            except TypeError:
                # 먼저 mode, 안되면 None exo, 마지막으로 None+mode
                try:
                    y_full = self.model(x, mode=(self.lmm_mode or "eval"))
                except TypeError:
                    try:
                        y_full = self.model(x, future_exo=None)
                    except TypeError:
                        y_full = self.model(x, future_exo=None, mode=(self.lmm_mode or "eval"))
        return self._normalize_out(y_full, B)

    def _try_model_call(self, x, future_exo, B):
        """(x, future_exo, mode) → (x, future_exo) → (x, mode) → (x) 순서로 안전 호출."""
        try:
            return self.model(x, future_exo=future_exo, mode=(self.lmm_mode or "eval"))
        except TypeError:
            pass
        try:
            return self.model(x, future_exo=future_exo)
        except TypeError:
            pass
        try:
            return self.model(x, mode=(self.lmm_mode or "eval"))
        except TypeError:
            pass
        return self.model(x)

    def _call_with_exo(self, x: torch.Tensor, B: int, Hm: int, step_offset: int) -> torch.Tensor:
        """미래 exo 콜백이 있으면 (B,Hm,exo_dim)로 만들어 모델에 주입 후 [B,Hm]로 정규화."""
        if self.future_exo_cb is None:
            return self._call_model(x, B)
        t0 = self.global_t0 + step_offset
        exo = self.future_exo_cb(t0, Hm)              # (Hm, exo_dim)
        exo = exo.to(x.device).unsqueeze(0).expand(B, -1, -1)  # (B,Hm,exo_dim)
        y_full = self._try_model_call(x, exo, B)
        return self._normalize_out(y_full, B)

    # ---------- Overlap-DMS ----------
    @torch.no_grad()
    def forecast_overlap_avg(self,
                             x_init: torch.Tensor,
                             horizon: int,
                             block_stride: int = 1,
                             device: Optional[torch.device] = None,
                             context_policy: str = 'once',
                             # 안정화 토글 & 파라미터
                             use_winsor: bool = True,
                             use_multi_guard: bool = True,
                             use_dampen: bool = True,
                             winsor_q: tuple = (0.05, 0.95),
                             winsor_mul: float = 2.0,
                             winsor_growth: float = 1.10,
                             max_step_up: float = 0.10,
                             max_step_down: float = 0.30,
                             damp_min: float = 0.2,
                             damp_max: float = 0.6) -> torch.Tensor:
        """
        Overlapped DMS averaging:
        - horizon 길이 만큼을 목표로, DMS 블록을 겹치게 예측하여 동일 시점 평균.
        - 각 블록 시작 → Hm 길이의 예측(y_block) 생성 → 해당 구간 누적/카운트
        - 다음 블록으로 전진할 때는 y_block의 앞쪽 'advance'개(= stride) 1-step 값을
          순차적으로 사용해 입력 윈도우를 업데이트(옵션 가드/댐핑 적용).
        """
        was_training = self.model.training
        self.model.eval()

        device = device or next(self.model.parameters()).device
        x = x_init.to(device).float().clone()
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B,L] -> [B,L,1]
        B = x.size(0)

        # TTM context
        if (self.ttm is not None) and (context_policy in ('once', 'per_step')):
            if context_policy == 'once':
                self.ttm.add_context(self._context_features(x))

        # --- Hm probe (exo 필수 모델도 안전하게) ---
        try:
            Hm = self._call_model(x, B).size(1)
        except Exception:
            # exo가 필요한 모델일 수 있으므로 exo 포함 호출로 추정
            H_guess = int(getattr(self.model, 'horizon',
                                  getattr(self.model, 'output_horizon', 120)))
            Hm = self._call_with_exo(x, B, H_guess, step_offset=0).size(1)

        H = int(horizon)
        stride = max(1, int(block_stride))

        # 첫 블록(미래 exo 포함)
        y_block0 = self._call_with_exo(x, B, Hm, step_offset=0)  # [B,Hm]

        pred_sum = torch.zeros(B, H, device=device)
        pred_cnt = torch.zeros(B, H, device=device)

        # 블록 시작 인덱스들(0..H-Hm, stride 간격) — H<Hm이면 starts=[0]
        if H >= Hm:
            starts = list(range(0, H - Hm + 1, stride))
        else:
            starts = [0]

        # Block loop
        for bi, b_start in enumerate(starts):
            if (self.ttm is not None) and (context_policy == 'per_step'):
                self.ttm.add_context(self._context_features(x))

            # 첫 블록 재활용, 이후는 step_offset=b_start로 DMS 호출
            y_block = y_block0 if bi == 0 else self._call_with_exo(x, B, Hm, step_offset=b_start)

            # 누적/카운트
            max_t = min(Hm, H - b_start)
            pred_sum[:, b_start:b_start + max_t] += y_block[:, :max_t]
            pred_cnt[:, b_start:b_start + max_t] += 1.0

            # 다음 블록을 위해 입력 윈도우 전진
            if bi < len(starts) - 1:
                advance = starts[bi + 1] - starts[bi]  # 보폭(= stride)
                advance = max(1, advance)

                # y_block의 앞쪽 advance개를 순서대로 사용해 한 스텝씩 전진
                for s in range(advance):
                    hist = x[:, :, self.target_channel]  # [B,L]
                    last = hist[:, -1]
                    y_step = y_block[:, s]  # 해당 시점의 1-step 예측

                    if use_winsor:
                        y_step = _winsorize_clamp(hist, y_step,
                                                  nonneg=True, clip_q=winsor_q,
                                                  clip_mul=winsor_mul, max_growth=winsor_growth)
                    if use_multi_guard:
                        y_step = _guard_multiplicative(last, y_step,
                                                       max_step_up=max_step_up,
                                                       max_step_down=max_step_down)
                    if use_dampen:
                        # 블록 진행도에 따른 가중(선형): bi 기준으로 완만히 증가
                        w = float(damp_min + (damp_max - damp_min) *
                                  (bi / max(1, len(starts) - 1)))
                        y_step = _dampen_to_last(last, y_step, damp=w)

                    x = _prepare_next_input(x, y_step,
                                            target_channel=self.target_channel,
                                            fill_mode=self.fill_mode)

        y_hat = pred_sum / torch.clamp(pred_cnt, min=1.0)

        if was_training:
            self.model.train()
        return y_hat

    # ---------- DMS + (필요시) IMS 확장 ----------
    @torch.no_grad()
    def forecast_DMS_to_IMS(self,
                            x_init: torch.Tensor,
                            horizon: Optional[int] = None,
                            device: Optional[torch.device] = None,
                            extend: str = "ims",  # "ims" | "error"
                            context_policy: str = "once",
                            y_true: Optional[torch.Tensor] = None,
                            teacher_forcing_ratio: float = 0.0,
                            # 안정화 토글 & 파라미터
                            use_winsor: bool = True,
                            use_multi_guard: bool = False,  # DMS 본구간은 보통 끔(원하면 켜기)
                            use_dampen: bool = True,
                            winsor_q: tuple = (0.05, 0.95),
                            winsor_mul: float = 4.0,
                            winsor_growth: float = 3.0,
                            max_step_up: float = 0.10,
                            max_step_down: float = 0.40,
                            damp: float = 0.5) -> torch.Tensor:
        """
        DMS 한 번으로 Hm까지 생성하고, horizon>Hm 이면 IMS로 이어붙임.
        future_exo_cb가 필요한 모델도 안전하게 Hm을 추정해 동작.
        """
        was_training = self.model.training
        self.model.eval()

        device = device or next(self.model.parameters()).device
        x = x_init.to(device).float().clone()
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B,L] -> [B,L,1]
        B = x.size(0)

        # 0) TTM context
        if (self.ttm is not None) and (context_policy in ("once", "per_step")):
            if context_policy == "once":
                self.ttm.add_context(self._context_features(x))

        # 1) Hm probe (exo 없이 먼저 시도, 실패 시 exo 포함 시도로 백업)
        def _probe_hm_safe() -> int:
            try:
                return self._call_model(x, B).size(1)
            except Exception:
                H_guess = int(getattr(self.model, "horizon",
                                      getattr(self.model, "output_horizon", 120)))
                return self._call_with_exo(x, B, H_guess, step_offset=0).size(1)

        Hm = _probe_hm_safe()
        H = Hm if horizon is None else int(horizon)

        # 2) DMS 1회 (미래 exo 포함) → y_block:[B,Hm]
        y_block = self._call_with_exo(x, B, Hm, step_offset=0)

        outputs = []
        use_tf = (y_true is not None) and (teacher_forcing_ratio > 0.0)
        if use_tf:
            y_true = y_true.to(device).float()

        # 3) 본구간: min(Hm, H) 만큼 스텝별 취득(안정화/TF 적용)
        use_len = min(Hm, H)
        for t in range(use_len):
            if (self.ttm is not None) and (context_policy == "per_step"):
                self.ttm.add_context(self._context_features(x))

            if use_tf and (t < (y_true.shape[1] if y_true.dim() > 1 else 0)) and \
                    (torch.rand(1).item() < teacher_forcing_ratio):
                y_step = y_true[:, t]
            else:
                hist = x[:, :, self.target_channel]
                last = hist[:, -1]
                y_step = y_block[:, t]

                if use_winsor:
                    y_step = _winsorize_clamp(hist, y_step,
                                              nonneg=True, clip_q=winsor_q,
                                              clip_mul=winsor_mul, max_growth=winsor_growth)
                if use_multi_guard:
                    y_step = _guard_multiplicative(last, y_step,
                                                   max_step_up=max_step_up,
                                                   max_step_down=max_step_down)
                if use_dampen:
                    y_step = _dampen_to_last(last, y_step, damp=damp)

            outputs.append(y_step.unsqueeze(1))
            x = _prepare_next_input(x, y_step,
                                    target_channel=self.target_channel,
                                    fill_mode=self.fill_mode)

        # 4) H <= Hm 이면 바로 반환
        if H <= Hm:
            y_hat = torch.cat(outputs, dim=1)  # [B,H]
            if was_training:
                self.model.train()
            return y_hat

        # 5) H > Hm: IMS로 초과 구간 생성
        if extend not in ("ims", "error"):
            raise ValueError("extend must be 'ims' or 'error'")
        if extend == "error":
            raise ValueError(f"horizon ({H}) > model_output ({Hm}). "
                             f"Set extend='ims' to extend autoregressively.")

        remaining = H - use_len
        for t in range(remaining):
            if (self.ttm is not None) and (context_policy == "per_step"):
                self.ttm.add_context(self._context_features(x))

            # step_offset = 이미 생성한 길이(use_len) + IMS의 상대 step t
            y_full = self._call_with_exo(x, B, Hm, step_offset=(use_len + t))
            y_raw = y_full[:, 0]

            hist = x[:, :, self.target_channel]
            last = hist[:, -1]
            y_step = y_raw

            if use_winsor:
                y_step = _winsorize_clamp(hist, y_step,
                                          nonneg=True, clip_q=winsor_q,
                                          clip_mul=winsor_mul, max_growth=winsor_growth)
            if use_multi_guard:
                y_step = _guard_multiplicative(last, y_step,
                                               max_step_up=max_step_up,
                                               max_step_down=max_step_down)
            if use_dampen:
                y_step = _dampen_to_last(last, y_step, damp=damp)

            outputs.append(y_step.unsqueeze(1))
            x = _prepare_next_input(x, y_step,
                                    target_channel=self.target_channel,
                                    fill_mode=self.fill_mode)

        y_hat = torch.cat(outputs, dim=1)  # [B,H]

        if was_training:
            self.model.train()
        return y_hat


class IMSForecaster:
    """
    Titan/LMM 계열을 위한 IMS(autoregressive) 예측기 (+미래 exo 주입/TTM/안정화 토글).
    - future_exo_cb(start_idx:int, H:int) -> Tensor(H, exo_dim)
    - global_t0: 예측 시작의 '절대 인덱스' (파트/시계열마다 다르게 세팅 권장)
    """

    def __init__(self,
                 model: torch.nn.Module,
                 target_channel: int = 0,
                 fill_mode: str = "copy_last",
                 lmm_mode: Optional[str] = None,
                 predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 ttm: Optional[object] = None,
                 future_exo_cb: Optional[Callable[[int, int], torch.Tensor]] = lambda s, h: make_calendar_exo(s, h, period=12)):
        self.model = model
        self.target_channel = target_channel
        self.fill_mode = fill_mode
        self.lmm_mode = lmm_mode
        self.predict_fn = predict_fn
        self.ttm = ttm

        # 달력/외생 피처 콜백 (예: make_calendar_exo)
        self.future_exo_cb = future_exo_cb
        self.global_t0 = 0  # 호출 전에 "예측 시작 절대 인덱스"로 세팅하세요.

    @torch.no_grad()
    def _context_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "input_proj"):
            return self.model.encoder.input_proj(x)
        return x

    def warmup(self,
               x_warm: torch.Tensor,
               y_warm: Optional[torch.Tensor] = None,
               steps: int = 1,
               add_context: bool = True,
               adapt: bool = True) -> None:
        if self.ttm is None:
            return
        if add_context:
            x_ctx = self._context_features(x_warm)
            self.ttm.add_context(x_ctx)
            if adapt and (y_warm is not None):
                _ = self.ttm.adapt(x_warm, y_warm, steps=steps)

    def _normalize_out(self, y_full: torch.Tensor, B: int) -> torch.Tensor:
        # 출력 형상 정규화 → [B, H]
        if y_full.dim() == 1:
            y_full = y_full.view(B, -1)
        elif y_full.dim() == 2:
            pass
        elif y_full.dim() == 3:
            if y_full.size(1) == 1:
                y_full = y_full[:, 0, :]
            elif y_full.size(-1) == 1:
                y_full = y_full[:, :, 0]
            else:
                y_full = y_full.reshape(B, -1)
        else:
            y_full = y_full.reshape(B, -1)
        return y_full

    def _call_model(self, x: torch.Tensor, B: int) -> torch.Tensor:
        if self.predict_fn is not None:
            y_full = self.predict_fn(x)
        else:
            try:
                y_full = self.model(x)
            except TypeError:
                y_full = self.model(x, mode=(self.lmm_mode or 'eval'))
        return self._normalize_out(y_full, B)

    def _try_model_call(self, x, future_exo, B):
        """
        모델이 future_exo / mode를 어떤 시그니처로 받는지 몰라도 안전하게 시도.
        (x, future_exo, mode) → (x, future_exo) → (x, mode) → (x)
        """
        try:
            return self.model(x, future_exo=future_exo, mode=(self.lmm_mode or "eval"))
        except TypeError:
            pass
        try:
            return self.model(x, future_exo=future_exo)
        except TypeError:
            pass
        try:
            return self.model(x, mode=(self.lmm_mode or "eval"))
        except TypeError:
            pass
        return self.model(x)

    def _call_with_exo(self, x: torch.Tensor, B: int, Hm: int, step_offset: int) -> torch.Tensor:
        """
        Hm 길이의 예측을 하되, future_exo_cb가 있으면 step_offset부터의 exo를 주입.
        """
        if self.future_exo_cb is None:
            return self._call_model(x, B)

        # 미래 외생 피처 만들기
        t0 = self.global_t0 + step_offset
        exo = self.future_exo_cb(t0, Hm)  # (Hm, exo_dim)
        exo = exo.to(x.device).unsqueeze(0).expand(B, -1, -1)  # (B,Hm,exo_dim)

        y_full = self._try_model_call(x, exo, B)
        return self._normalize_out(y_full, B)

    @torch.no_grad()
    def forecast(self,
                 x_init: torch.Tensor,
                 horizon: int,
                 device: Optional[torch.device] = None,
                 y_true: Optional[torch.Tensor] = None,
                 teacher_forcing_ratio: float = 0.0,
                 context_policy: str = 'once',  # 'none' | 'once' | 'per_step'
                 # 안정화 토글
                 use_winsor: bool = True,
                 use_multi_guard: bool = True,
                 use_dampen: bool = True,
                 winsor_q: tuple = (0.05, 0.95),
                 winsor_mul: float = 4.0,
                 winsor_growth: float = 3.0,
                 max_step_up: float = 0.10,
                 max_step_down: float = 0.30,
                 damp: float = 0.5
                 ) -> torch.Tensor:
        """
        Returns
        - y_hat: [B, horizon]
        """
        was_training = self.model.training
        self.model.eval()

        device = device or next(self.model.parameters()).device
        x = x_init.to(device).float().clone()
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        B = x.size(0)

        # TTM: 시작 시점 메모리 주입
        if (self.ttm is not None) and (context_policy in ('once', 'per_step')):
            if context_policy == 'once':
                x_ctx = self._context_features(x)
                self.ttm.add_context(x_ctx)

        # 모델의 출력 길이(Hm) probe (exo 없이 한 번)
        y_probe = self._call_model(x, B)      # [B, Hm]
        Hm = y_probe.size(1)

        outputs = []
        use_tf = (y_true is not None) and (teacher_forcing_ratio > 0.0)
        if use_tf:
            y_true = y_true.to(device).float()

        for t in range(horizon):

            # per-step TTM
            if (self.ttm is not None) and (context_policy == 'per_step'):
                x_ctx = self._context_features(x)
                self.ttm.add_context(x_ctx)

            # 1) 모델 호출 (미래 exo 포함)
            y_full = self._call_with_exo(x, B, Hm, step_offset=t)  # [B, Hm]
            y_raw = y_full[:, 0]                                   # 첫 스텝만 사용

            # 2) 한 스텝 결정: TF or 안정화 예측
            if use_tf and (t < y_true.shape[1]) and (torch.rand(1).item() < teacher_forcing_ratio):
                y_step = y_true[:, t]
            else:
                hist = x[:, :, self.target_channel]   # [B, L]
                last = hist[:, -1]

                y_step = y_raw
                if use_winsor:
                    y_step = _winsorize_clamp(hist, y_step,
                                              nonneg=True, clip_q=winsor_q,
                                              clip_mul=winsor_mul, max_growth=winsor_growth)
                if use_multi_guard:
                    y_step = _guard_multiplicative(last, y_step,
                                                  max_step_up=max_step_up,
                                                  max_step_down=max_step_down)
                if use_dampen:
                    y_step = _dampen_to_last(last, y_step, damp=damp)

            outputs.append(y_step.unsqueeze(1))  # [B,1]

            # 3) 입력 윈도우 업데이트
            x = _prepare_next_input(x, y_step,
                                    target_channel=self.target_channel,
                                    fill_mode=self.fill_mode)

        y_hat = torch.cat(outputs, dim=1)  # [B, horizon]

        if was_training:
            self.model.train()
        return y_hat


def evaluate(model, loader, device='cpu'):
    model.to(device)
    model.eval()

    preds = []
    part_ids = []

    with torch.no_grad():
        for batch in loader:
            x, part = batch  # inference에서는 y가 없을 수 있음
            x = x.to(device)

            pred = model(x)
            # pred = model.revin_layer(pred, 'denorm')
            preds.append(pred.cpu())
            # trues는 없음 → 추론 모드에서는 실제 y를 모를 수 있으니 skip
            part_ids.extend(part)

    all_preds = torch.cat(preds, dim=0)
    return all_preds, part_ids


def evaluate_with_truth(model, loader, device='cpu'):
    model.to(device)
    model.eval()

    preds, trues, part_ids = [], [], []

    with torch.no_grad():
        for batch in loader:
            x, y, part = batch
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred = pred.clone()  # ✅ 복사
            # pred = model.revin_layer(pred, 'denorm')
            pred = pred.reshape(pred.shape[0], -1, 1).cpu()

            y = y.reshape(y.shape[0], -1, 1).cpu()

            preds.append(pred)
            trues.append(y)
            part_ids.extend(part)

    return torch.cat(preds, dim=0), torch.cat(trues, dim=0), part_ids
