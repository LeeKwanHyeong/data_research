from typing import Optional, Callable
import torch
from utils.inference_utils import _winsorize_clamp, _guard_multiplicative, _dampen_to_last, _prepare_next_input


class DMSForecaster:
    """
    Titan/LMM 계열 모델을 위한 DMS(Direct Multi-Step) 예측기 + TTM/안정화 지원.
    - 기본: model(x) 1회 호출로 [B, H_model] 출력 사용
    - horizon < H_model: 앞에서부터 슬라이스
    - horizon > H_model: extend='ims' 설정 시 초과 구간은 IMS(한 스텝씩)로 확장
                         extend='error' 설정 시 예외 발생
    - TTM(Test-Time Memory) 옵션: context_policy ('none'|'once'|'per_step')
    - 안정화: 각 스텝 y_step에 winsorize + damp 적용
    - Teacher Forcing: 검증/백테스트 시 y_true + teacher_forcing_ratio 사용 가능
    """
    def __init__(self,
                 model: torch.nn.Module,
                 target_channel: int = 0,
                 fill_mode: str = "copy_last",
                 lmm_mode: Optional[str] = None,
                 predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 ttm: Optional[object] = None):
        self.model = model
        self.target_channel = target_channel
        self.fill_mode = fill_mode
        self.lmm_mode = lmm_mode
        self.predict_fn = predict_fn
        self.ttm = ttm

    @torch.no_grad()
    def _context_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        컨텍스트 메모리에 저장할 특징을 생성.
        - 기본: encoder.input_proj(x) → [B, L, d_model]
        - 없으면 x 그대로 (단, 이 경우 차원 일치 책임은 호출 측)
        """
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "input_proj"):
            return self.model.encoder.input_proj(x)
        return x

    def warmup(self,
               x_warm: torch.Tensor,
               y_warm: Optional[torch.Tensor] = None,
               steps: int = 1,
               add_context: bool = True,
               adapt: bool = True) -> None:
        """
        x_warm, y_warm을 이용해 TTM으로 사전 적응.
        - add_context: contextual memory만 업데이트
        - adapt: y_warm이 있을 때 파라미터를 소폭 업데이트(gradient 기반)
        """
        if self.ttm is None:
            return
        if add_context:
            x_ctx = self._context_features(x_warm)
            self.ttm.add_context(x_ctx)
        if adapt and (y_warm is not None):
            _ = self.ttm.adapt(x_warm, y_warm, steps=steps)

    def _call_model(self, x: torch.Tensor, B: int) -> torch.Tensor:
        if self.predict_fn is not None:
            y_full = self.predict_fn(x)
        else:
            try:
                y_full = self.model(x)
            except TypeError:
                # e.g., LMMModel(x, mode='eval')
                y_full = self.model(x, mode=(self.lmm_mode or "eval"))

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

    @torch.no_grad()
    def forecast_overlap_avg(self,
                             x_init: torch.Tensor,
                             horizon: int,
                             device:torch.device | str | None = None,
                             context_policy: str = 'once',
                             # --안정화 가드 옵션 (윈도우 슬라이드에 쓰일 1-step 업데이트용 --
                             use_winsor: bool = True,
                             use_multi_guard: bool = True,
                             use_dampen: bool = True,
                             clip_q: tuple[float, float] = (0.05, 0.95),
                             clip_mul: float = 2.0, # 분위수 상한 배수 (보수적)
                             max_growth: float = 1.10, # 직전값의 110% 상한
                             max_step_up: float = 0.10, # +10%/step
                             max_step_down: float = 0.30, # -30%/step
                             damp_min: float = 0.2, # IMS tail 초반 혼합 가중
                             damp_max: float = 0.6 # IMS tail 후반 혼합 가중
                        ) -> torch.Tensor:
        """
        Overlapped DMS averaging:
        - 블록을 겹치게 여러 번 예측하고 동일 시점의 값들을 평균.
        - 윈도우 슬라이드(다음 블록 시작)에는 '첫 스텝' 예측을 사용하며,
          필요시 winsorize / multiplicative guard / damp를 적용해 안정화
        """
        was_training = self.model.training
        self.model.eval()

        device = device or next(self.model.parameters()).device
        x = x_init.to(device).float().clone()
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B = x.size(0)

        # TTM Context 주입
        if (self.ttm is not None) and (context_policy in ('once', 'per_step')):
            if context_policy == 'once':
                x_ctx = self._context_features(x)
                self.ttm.add_context(x_ctx)

        # 첫 블록으로 Hm 파악
        y_block0 = self._call_model(x, B) # [B, Hm]
        Hm = y_block0.size(1)
        H = int(horizon)

        # 누적 합/카운트 버퍼
        pred_sum = torch.zeros(B, H, device = device)
        pred_cnt = torch.zeros(B, H, device = device)

        # 블록 개수: 마지막 시작 인덱스 = H - Hm
        num_blocks = max(1, H - Hm+1)

        # Block Loop
        for bidx in range(num_blocks):
            # per-step Context
            if (self.ttm is not None) and (context_policy == 'per_step'):
                x_ctx = self._context_features(x)
                self.ttm.add_context(x_ctx)

            if bidx == 0:
                y_block = y_block0 # 첫 블록은 재사용
            else:
                y_block = self._call_model(x, B) # [B, Hm]

            # 이번 블록이 기여할 전역 구간
            # 글로벌 인덱스 g = bidx + t(0-based), g < H
            max_t = min(Hm, H - bidx)
            pred_sum[:, bidx:bidx+max_t] += y_block[:, :max_t]
            pred_cnt[:, bidx:bidx+max_t] += 1.0

            # 다음 블록을 위한 윈도우 슬라이드 (첫 스텝 예측 사용)
            if bidx < num_blocks - 1:
                y_raw = y_block[:, 0] # [B]
                hist = x[:, :, self.target_channel] # [B, L]
                last = hist[:, -1] # [B]

                y_step = y_raw

                # (1) 절대치 가드 (winsorize)
                if use_winsor and ('_winsorize_clamp' in globals()):
                    y_step = _winsorize_clamp(
                        hist, y_step,
                        nonneg = True, clip_q = clip_q,
                        clip_mul = clip_mul, max_growth = max_growth
                    )

                # (2) 배수 변화 가드
                if use_multi_guard and ('_guard_multiplicative' in globals()):
                    y_step = _guard_multiplicative(
                        last, y_step,
                        max_step_up = max_step_up, max_step_down = max_step_down
                    )

                # (3) 평활
                if use_dampen:
                    w = float(damp_min + (damp_max - damp_min) * (bidx / max(1, num_blocks - 1)))
                    y_step = _dampen_to_last(last, y_step, damp=w) if ('_dampen_to_last' in globals()) else y_step

                x = _prepare_next_input(
                    x, y_step,
                    target_channel = self.target_channel,
                    fill_mode = self.fill_mode
                )

        y_hat = pred_sum / torch.clamp(pred_cnt, min = 1.0)

        if was_training:
            self.model.train()
        return y_hat

    @torch.no_grad()
    def forecast(self,
                 x_init: torch.Tensor,
                 horizon: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 extend: str = "ims",                # "ims" | "error"
                 context_policy: str = "once",       # "none" | "once" | "per_step"
                 y_true: Optional[torch.Tensor] = None,
                 teacher_forcing_ratio: float = 0.0  # 검증/백테스트용
                 ) -> torch.Tensor:
        """
        Args
        ----
        x_init : [B, L, C] (또는 [B, L] → 내부에서 [B, L, 1]로 보정)
        horizon : 원하는 예측 길이. None이면 모델 출력 길이로 자동 설정
        extend : horizon > H_model일 때의 처리 방식
                 - "ims": 초과 구간을 IMS로 이어붙임 (권장)
                 - "error": 예외 발생
        context_policy : TTM 컨텍스트 주입 정책 ("none"|"once"|"per_step")
        y_true / teacher_forcing_ratio : 검증에서만 사용 권장

        Returns
        -------
        y_hat : [B, horizon]
        """
        was_training = self.model.training
        self.model.eval()

        device = device or next(self.model.parameters()).device
        x = x_init.to(device).float().clone()
        if x.dim() == 2:                  # [B, L] → [B, L, 1]
            x = x.unsqueeze(-1)
        B = x.size(0)

        # TTM: 시작 시점 메모리 주입
        if (self.ttm is not None) and (context_policy in ("once", "per_step")):
            if context_policy == "once":
                x_ctx = self._context_features(x)
                self.ttm.add_context(x_ctx)

        # 1) DMS 1회 호출
        y_block = self._call_model(x, B)  # [B, H_model]
        Hm = y_block.size(1)

        # horizon 기본값: 모델 출력 길이
        H = Hm if horizon is None else int(horizon)

        outputs = []
        use_tf = (y_true is not None) and (teacher_forcing_ratio > 0.0)
        if use_tf:
            y_true = y_true.to(device).float()

        # 2) 요청한 horizon이 모델 출력 이내이면, 스텝별로 안정화 적용하며 슬라이스
        use_len = min(Hm, H)
        for t in range(use_len):
            # TTM per-step 옵션
            if (self.ttm is not None) and (context_policy == "per_step"):
                x_ctx = self._context_features(x)
                self.ttm.add_context(x_ctx)

            # 한 스텝 선택: 예측 or teacher forcing
            if use_tf and (t < (y_true.shape[1] if y_true.dim() > 1 else 0)) and \
               (torch.rand(1).item() < teacher_forcing_ratio):
                y_step = y_true[:, t]
            else:
                # 안정화 적용
                hist = x[:, :, self.target_channel]  # [B, L]
                y_step_raw = y_block[:, t]          # [B]
                y_step = _winsorize_clamp(hist, y_step_raw,
                                          nonneg=True, clip_q=(0.05, 0.95),
                                          clip_mul=4.0, max_growth=3.0)
                y_step = _dampen_to_last(hist[:, -1], y_step, damp=0.5)

            outputs.append(y_step.unsqueeze(1))
            x = _prepare_next_input(
                x, y_step,
                target_channel=self.target_channel,
                fill_mode=self.fill_mode
            )

        # 3) horizon 이내면 바로 반환
        if H <= Hm:
            y_hat = torch.cat(outputs, dim=1)  # [B, H]
            if was_training:
                self.model.train()
            return y_hat

        # 4) horizon 초과: 전략에 따라 처리
        if extend not in ("ims", "error"):
            raise ValueError(f"extend must be 'ims' or 'error', got {extend}")
        if extend == "error":
            raise ValueError(
                f"horizon ({H}) > model_output ({Hm}). "
                f"Set extend='ims' to extend autoregressively."
            )

        # 5) extend == "ims": 초과 구간을 한 스텝씩 생성하며 입력에 붙여가며 예측
        remaining = H - use_len

        for t in range(remaining):
            if (self.ttm is not None) and (context_policy == "per_step"):
                x_ctx = self._context_features(x)
                self.ttm.add_context(x_ctx)

            y_block = self._call_model(x, B)  # [B, H_model]
            y_raw = y_block[:, 0]  # 첫 스텝만 사용

            hist = x[:, :, self.target_channel]  # [B, L]
            last = hist[:, -1]

            # 절대치 가드(분위수/직전)
            y_step = _winsorize_clamp(
                hist, y_raw,
                nonneg=True, clip_q=(0.05, 0.95),
                clip_mul=1.0,  # 더 보수적으로 (기존 4.0 → 2.0 권장)
                max_growth=1.0  # 직전값의 110% 상한 (배수 해석 시 1.10 권장)
            )
            # 곱셈 가드(배수 변화)
            y_step = _guard_multiplicative(
                last, y_step,
                max_step_up=0.10,  # +10%/step
                max_step_down=0.40  # -30%/step
            )
            # ③ 평활 (스텝별 가중 증가 스케줄 권장)
            #    예: 초반은 예측 반영을 낮게, 후반으로 갈수록 조금씩 키움
            damp_min, damp_max = 0.2, 0.3
            w = damp_min + (damp_max - damp_min) * (t / max(1, remaining - 1))
            y_step = _dampen_to_last(last, y_step, damp=float(w))

            outputs.append(y_step.unsqueeze(1))
            x = _prepare_next_input(
                x, y_step,
                target_channel=self.target_channel,
                fill_mode=self.fill_mode
            )

        y_hat = torch.cat(outputs, dim=1)  # [B, H]

        if was_training:
            self.model.train()
        return y_hat




class IMSForecaster:
    """
        Titan/LMM 계열을 위한 IMS(autoregressive) 예측기.
        - 함수 버전과 동일 로직 + 설정/옵션을 인스턴스에 유지
        - 필요 시 Teacher Forcing, predict_fn, TTA 연동 등의 확장 지점 제공
    """
    def __init__(self,
                 model: torch.nn.Module,
                 target_channel: int = 0,
                 fill_mode: str = "copy_last",
                 lmm_mode: Optional[str] = None,
                 predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 ttm: Optional[object] = None,
                 ):
        self.model = model
        self.target_channel = target_channel
        self.fill_mode = fill_mode
        self.lmm_mode = lmm_mode
        self.predict_fn = predict_fn
        self.ttm = ttm

    @torch.no_grad()
    def _context_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        컨텍스트 메모리에 저장할 특징을 생성.
        - 기본: encoder.input_proj(x) → [B, L, d_model]
        - 없으면 x 그대로 (단, 이 경우 차원 일치 책임은 호출 측)
        """
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "input_proj"):
            # x는 [B, L, C]여야 합니다.
            return self.model.encoder.input_proj(x)
        return x

    def warmup(self,
               x_warm: torch.Tensor,
               y_warm: Optional[torch.Tensor] = None,
               steps : int = 1,
               add_context: bool = True,
               adapt: bool = True) -> None:
        '''
            x_warm, y_warm을 이용해 TTM으로 사전 적응.
            - add_context: contextual memory만 업데이트
            - adapt: y_warm이 있을 때 파라미터를 소폭 업데이트(Gradient Based)
        '''
        if self.ttm is None:
            return
        if add_context:
            x_ctx = self._context_features(x_warm)
            self.ttm.add_context(x_ctx)
            if adapt and (y_warm is not None):
                # ttm.adapt 내부에서 model(x_warm)과 y_warm 간 MSE로 파라미터 업데이트
                _ = self.ttm.adapt(x_warm, y_warm, steps = steps)

    def _call_model(self, x: torch.Tensor, B: int) -> torch.Tensor:
        if self.predict_fn is not None:
            y_full = self.predict_fn(x)
        else:
            try:
                y_full = self.model(x)
            except TypeError:
                y_full = self.model(x, mode = (self.lmm_mode or 'eval'))

        # 출력 정규화 -> [B, H]
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

    '''
        기존 함수처럼 간단 호출이 필요하면: 함수 유지(현재 코드 OK)
        y_hat = autoregressive_forecast(model, x_init, horizon=120, lmm_mode='eval')
        
        # 옵션/상태를 재사용하고 싶으면: 클래스 사용
        forecaster = IMSForecaster(model, target_channel=0, fill_mode="copy_last", lmm_mode="eval")
        y_hat = forecaster.forecast(x_init, horizon=120)
        
        # 검증 단계에서 Teacher Forcing 일부 적용(안정화용)
        y_hat_tf = forecaster.forecast(x_init, horizon=120, y_true=y_val, teacher_forcing_ratio=0.2)
    '''

    @torch.no_grad()
    def forecast(self,
                 x_init: torch.Tensor,
                 horizon: int,
                 device: Optional[torch.device] = None,
                 y_true: Optional[torch.Tensor] = None,
                 teacher_forcing_ratio: float = 0.0,
                 context_policy: str = 'once' # 'none' | 'once' | 'per_step'
                 ) -> torch.Tensor:
        """
        Args
        - x_init: [B, L, C]
        - horizon: 생성할 스텝 수
        - y_true: [B, H] (있으면 teacher forcing에 사용)
        - teacher_forcing_ratio: 0.0~1.0 (각 스텝에서 y_true 사용 확률)

        Returns
        - y_hat: [B, horizon]
        """
        was_training = self.model.training
        self.model.eval()

        device = device or next(self.model.parameters()).device
        x = x_init.to(device).float().clone()
        if x.dim() == 2:
            x = x.unsqueeze(-1) # [B, L] -> [B, L, 1]
        B = x.size(0)

        # TTM: 시작 시점 메모리 주입
        if (self.ttm is not None) and (context_policy in ('once', 'per_step')):
            if context_policy == 'once':
                x_ctx = self._context_features(x)
                self.ttm.add_context(x_ctx)

        outputs = []
        use_tf = (y_true is not None) and (teacher_forcing_ratio > 0.0)
        if use_tf:
            y_true = y_true.to(device).float()


        for t in range(horizon):

            # 0) TTM: 스텝별 메모리 업데이트(optional)
            if (self.ttm is not None) and (context_policy == 'per_step'):
                x_ctx = self._context_features(x)
                self.ttm.add_context(x_ctx)

            # 1) 모델 호출
            y_full = self._call_model(x, B)

            # 3) 한 스텝 결정: 예측 or teacher forcing
            if use_tf and (t < y_true.shape[1]) and (torch.rand(1).item() < teacher_forcing_ratio):
                y_step = y_true[:, t]              # GT 사용
            else:
                # y_step = y_full[:, 0]              # 모델 예측 사용
                hist = x[:, :, self.target_channel]
                y_step = _winsorize_clamp(hist, y_full[:, 0],
                                          nonneg=True, clip_q = (0.05, 0.95),
                                          clip_mul = 4.0, max_growth=3.0)
                y_step = _dampen_to_last(hist[:, -1], y_step, damp = 0.5)

            outputs.append(y_step.unsqueeze(1))     # [B,1]

            # 4) 입력 윈도우 업데이트
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
