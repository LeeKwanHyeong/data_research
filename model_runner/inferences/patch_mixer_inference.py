from typing import Optional, Callable, Sequence, Tuple
import torch
from utils.inference_utils import _winsorize_clamp, _guard_multiplicative, _dampen_to_last, _prepare_next_input

# ---------- Common ----------
def _infer_quantiles(model) -> Optional[Tuple[float, ...]]:
    """Model quantile 정보가 있으면 가져오기. (없으면 None)"""
    try:
        if hasattr(model, 'head') and hasattr(model.head, 'qs'):
            qs = tuple(float(q) for q in model.head.qs)
            return qs
    except Exception:
        pass
    return None

def _pick_q_index(qs: Sequence[float], q_use: float) -> int:
    """가장 가까운 분위수 인덱스 선택"""
    import math
    return int(min(range(len(qs)), key = lambda i: math.fabs(qs[i] - q_use)))

def _to_BH(pred: torch.Tensor, B: int) -> torch.Tensor:
    """예측 출력을 (B, H)로 평탄화(점추정 모델용)"""
    if pred.dim() == 1:
        return pred.view(B, -1)
    if pred.dim() == 2:
        return pred
    if pred.dim() == 3:
        # (B, 1, H) or (B, H, 1) 같은 경우만 평탄화 허용
        if pred.size(1) == 1:
            return pred[:, 0, :]
        if pred.size(-1) == 1:
            return pred[:, :, 0]

    # 기타: 안전하게 평탄화
    return pred.reshape(B, -1)

# =======================================================
# DMS Forecaster (Overlap Averaging) — Quantile/Point 공통
# =======================================================
class PatchMixer_DMSForecaster:
    """
    PatchMixer 계열 모델용 DMS(Direct Multi-Step) 예측기.
    - 모델이 (B, Q, Hm) or (B, Hm) 출력 모두 지원
    - overlap 평균, 1-step slide 안정화 (winsor/multiplicative/dampen)
    - 분위수 모델일 때:
        * 슬리이드/안정화는 q_use(기본 0.5) 사용
        * return_full = True면 전 구간 (B, Q, H) 분위수 시퀀스 반환
          (overlap 평균도 Q 축에 대해 동일하게 수행)
    """
    def __init__(self,
                 model: torch.nn.Module,
                 target_channel: int = 0,
                 fill_mode: str = 'copy_last',
                 predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 ttm: Optional[object] = None,
                 qs: Optional[Tuple[float, ...]] = None, # (Option) 분위수 목록을 명시적으로 주입.
                 ):
        self.model = model
        self.target_channel = target_channel
        self.fill_mode = fill_mode
        self.predict_fn = predict_fn
        self.ttm = ttm
        self.qs = qs or _infer_quantiles(model)


    @torch.no_grad()
    def _context_features(self, x:torch.Tensor) -> torch.Tensor:
        # 필요시, PatchMixer Backbone 내부 특징으로 교체 가능 (현재는 입력 그대로..)
        return x

    def _call_model(self, x: torch.Tensor) -> torch.Tensor:
        if self.predict_fn is not None:
            return self.predict_fn(x)
        return self.model(x)

    @torch.no_grad()
    def forecast_overlap_avg(self,
                             x_init: torch.Tensor,
                             horizon: int,
                             device: Optional[torch.device] = None,
                             context_policy: str = 'once', # 'none'|'once'|'per_step'
                             # --- Stabilize ---
                             use_winsor: bool = True,
                             use_multi_guard: bool = True,
                             use_dampen: bool = True,
                             clip_q: Tuple[float, float] = (0.05, 0.95),
                             clip_mul: float = 2.0,
                             max_growth: float = 1.10,
                             max_step_up: float = 0.10,
                             max_step_down: float = 0.30,
                             damp_min: float = 0.2,
                             damp_max: float = 0.6,
                             # --- Quantile ---
                             q_use: float = 0.5,        # Slide/Stabilize에 사용할 median quantile
                             return_full: bool = False  # True면 (B, Q, H) 반환(Quantile 모델일 경우)
                             ) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()

        device = device or next(self.model.parameters()).device
        x = x_init.to(device).float().clone()
        if x.dim() == 2: # (B, L) -> (B, L, 1)
            x = x.unsqueeze(-1)
        B, L, _ = x.shape

        # TTM Context
        if (self.ttm is not None) and (context_policy in ('once', 'per_step')):
            if context_policy == 'once':
                self.ttm.add_context(self._context_features(x))


        # First Block
        y0 = self._call_model(x)

        Hm = y0.size(-1) if y0.dim() >= 2 else y0.numel() // B
        H = int(horizon)

        is_quantile = (y0.dim() == 3)   # (B, Q, Hm)
        if is_quantile:
            qs = self.qs or (0.1, 0.5, 0.9)
            q_idx = _pick_q_index(qs, q_use)

        # Cumulative Buffer
        if is_quantile and return_full:
            Q = y0.size(1)
            pred_sum_q = torch.zeros(B, Q, H, device = device)
            pred_cnt_q = torch.zeros(B, Q, H, device = device)
        else:
            pred_sum = torch.zeros(B, H, device = device)
            pred_cnt = torch.zeros(B, H, device = device)

        # Block num
        num_blocks = max(1, H - Hm + 1)

        for bidx in range(num_blocks):
            if (self.ttm is not None) and (context_policy == 'per_step'):
                self.ttm.add_context(self._context_features(x))

            # Current Block Inference
            y_blk = y0 if bidx == 0 else self._call_model(x)

            if is_quantile:
                # (B, Q, Hm)
                max_t = min(Hm, H - bidx)
                if return_full:
                    pred_sum_q[:, :, bidx:bidx+max_t] += y_blk[:, :, :max_t]
                    pred_cnt_q[:, :, bidx:bidx+max_t] += 1.0
                else:
                    pred_sum[:, bidx:bidx+max_t] += y_blk[:, q_idx, :max_t]
                    pred_cnt[:, bidx:bidx+max_t] += 1.0

                # Slide용 1-step은 0.5 Quantile
                y_raw = y_blk[:, q_idx, 0] # (B,)

            else:
                # (B, Hm) or Smoothing (B, Hm)
                y_blk = _to_BH(y_blk, B)
                max_t = min(Hm, H - bidx)
                pred_sum[:, bidx:bidx+max_t] += y_blk[:, :max_t]
                pred_cnt[:, bidx:bidx+max_t] += 1.0
                y_raw = y_blk[:, 0]

            # Next block window slide
            if bidx < num_blocks - 1:
                hist = x[:, :, self.target_channel] # (B, L)
                last = hist[:, -1]

                y_step = y_raw
                if use_winsor:
                    y_step = _winsorize_clamp(hist, y_step, nonneg=True, clip_q = clip_q,
                                              clip_mul = clip_mul, max_growth = max_growth)
                if use_multi_guard:
                    y_step = _guard_multiplicative(last, y_step, max_step_up = max_step_up,
                                                   max_step_down = max_step_down)
                if use_dampen:
                    w = float(damp_min + (damp_max - damp_min) * (bidx / max(1, num_blocks - 1)))
                    y_step = _dampen_to_last(last, y_step, damp = w)

                x = _prepare_next_input(x, y_step, target_channel = self.target_channel,
                                        fill_mode = self.fill_mode)
        if is_quantile and return_full:
            y_hat = pred_sum_q / torch.clamp(pred_cnt_q, min = 1.0)
        else:
            y_hat = pred_sum / torch.clamp(pred_cnt, min = 1.0)

        if was_training:
            self.model.train()
        return y_hat # (B, H) or (B, Q, H)

# ==============================================
# IMS Forecaster — Quantile/Point 공통
# ==============================================
class PatchMixer_IMSForecaster:
    """
    PatchMixer 계열용 IMS(autoregressive) 예측기.
    - 모델이 (B, Q, Hm) or (B, Hm) 출력 모두 지원
    - Quantile Model: Step update는 q_use 사용, return_full = True 면 (B, Q, H) 반환
      (각 스텝의 첫 번째 예측 위치 t=0의 분위수를 쌓아 H칸 생성)
    """
    def __init__(self,
                 model: torch.nn.Module,
                 target_channel: int = 0,
                 fill_mode: str = 'copy_last',
                 predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 ttm: Optional[object] = None,
                 qs: Optional[Tuple[float, ...]] = None
                 ):
        self.model = model
        self.target_channel = target_channel
        self.fill_mode = fill_mode
        self.predict_fn = predict_fn
        self.ttm = ttm
        self.qs = qs or _infer_quantiles(model)

    @torch.no_grad()
    def _context_features(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _call_model(self, x: torch.Tensor) -> torch.Tensor:
        if self.predict_fn is not None:
            return self.predict_fn(x)
        return self.model(x)

    @torch.no_grad()
    def forecast(self,
                 x_init: torch.Tensor,
                 horizon: int,
                 device: Optional[torch.device] = None,
                 context_policy: str = 'once',
                 q_use: float = 0.5,
                 return_full: bool = False,
                 # Stabilize option
                 use_winsor: bool = True,
                 use_dampen: bool = True,
                 clip_q: Tuple[float, float] = (0.05, 0.95),
                 clip_mul: float = 2.0,
                 max_growth: float = 1.10,
                 damp: float = 0.5,
                 ) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()

        device = device or next(self.model.parameters()).device
        x = x_init.to(device).float().clone()
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, L, _ = x.shape

        # TTM Context
        if (self.ttm is not None) and (context_policy in ('once', 'per_step')):
            if context_policy == 'once':
                self.ttm.add_context(self._context_features(x))

        is_quantile = False
        qs = self.qs or (0.1, 0.5, 0.9)
        q_idx = _pick_q_index(qs, q_use)

        if return_full:
            # 예측 전 1회 호출로 Q 크기 파악
            y_probe = self._call_model(x)
            if y_probe.dim() == 3: # (B, Q, Hm)
                is_quantile = True
                Q = y_probe.size(1)
                out_full = torch.zeros(B, Q, horizon, device = device)
            else:
                out_full = None # if point estimator then pass

        outputs = []
        for t in range(horizon):
            if (self.ttm is not None) and (context_policy == 'per_step'):
                self.ttm.add_context(self._context_features(x))

            y_full = self._call_model(x)

            if y_full.dim() == 3:
                is_quantile = True
                y_step_center = y_full[:, q_idx, 0] # (B,)
                if return_full:
                    out_full[:, :, t] = y_full[:, :, 0] # 각 스텝의 t=0 분위수 벡터 저장
            else:
                y_step_center = _to_BH(y_full, B)[:, 0] # (B,)

            hist = x[:, :, self.target_channel]
            if use_winsor:
                y_step = _winsorize_clamp(hist, y_step_center, nonneg = True, clip_q = clip_q,
                                          clip_mul = clip_mul, max_growth = max_growth)
            else:
                y_step = y_step_center

            if use_dampen:
                y_step = _dampen_to_last(hist[:, -1], y_step, damp = damp)

            outputs.append(y_step.unsqueeze(1))
            x = _prepare_next_input(x, y_step, target_channel = self.target_channel, fill_mode = self.fill_mode)

        y_hat_point = torch.cat(outputs, dim = 1) # (B, H)
        if was_training:
            self.model.train()

        if is_quantile and return_full:
            return out_full  # (B, Q, H)
        return y_hat_point   # (B, H)


# ==========================
# 평가 헬퍼
# ==========================
@torch.no_grad()
def evaluate_PatchMixer(model, loader, device = 'cpu', q_use: float = 0.5, return_full: bool = False):
    model.to(device)
    model.eval()

    preds, part_ids = [], []
    for batch in loader:
        # inference loader는 (x, part) or (x, y, part)일 수 있음
        if len(batch) == 2:
            x, part = batch
        else:
            x, _, part = batch
        x = x.to(device)

        pred = model(x)   # (B, H) or (B, Q, H)
        if pred.dim() == 3:
            if return_full:
                preds.append(pred.cpu())
            else:
                qs = _infer_quantiles(model) or (0.1, 0.5, 0.9)
                idx = _pick_q_index(qs, q_use)
                preds.append(pred[:, idx, :].cpu())
        else:
            preds.append(_to_BH(pred, pred.size(0)).cpu())

        part_ids.extend(part)

    all_preds = torch.cat(preds, dim = 0)
    return all_preds, part_ids # (B, H) or (B, Q, H), [part_id,...]

@torch.no_grad()
def evaluate_PatchMixer_with_truth(model, loader, device = 'cpu', q_use: float = 0.5, return_full: bool = False):
    model.to(device)
    model.eval()

    preds, trues, part_ids = [], [], []
    for batch in loader:
        x, y, part = batch
        x, y = x.to(device), y.to(device)

        pred = model(x)
        if pred.dim() == 3:
            if return_full:
                preds.append(pred.cpu())    # (B, Q, H)
            else:
                qs = _infer_quantiles(model) or (0.1, 0.5, 0.9)
                idx = _pick_q_index(qs, q_use)
                preds.append(pred[:, idx, :].cpu())         # (B, H)

        else:
            preds.append(_to_BH(pred, pred.size(0)).cpu())  # (B, H)

        trues.append(y.reshape(y.shape[0], -1, 1).cpu())    # (B, H, 1)
        part_ids.extend(part)

    return torch.cat(preds, dim = 0), torch.cat(trues, dim = 0), part_ids


