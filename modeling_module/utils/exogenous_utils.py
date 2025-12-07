import torch
import torch.nn as nn
from typing import Literal, Optional, Callable, List


def calendar_sin_cos(t: torch.Tensor, period: float) -> torch.Tensor:
    """
    단일 주기에 대한 sin/cos 쌍 반환 (..., 2)
    """
    return torch.stack([
        torch.sin(2 * torch.pi * t / period),
        torch.cos(2 * torch.pi * t / period)
    ], dim=-1)


def compose_exo_calendar_cb(date_type: str = "W", *, sincos: bool = True) -> Callable:
    """
    date_type에 따른 주기적 외생변수 생성 콜백 반환.

    지원 date_type:
      - 'M' (Monthly): 12개월 주기
      - 'W' (Weekly) : 52주 주기
      - 'D' (Daily)  : 7일(요일) + 365.25일(연간) 주기 (E=4 if sincos)
      - 'H' (Hourly) : 24시간(일간) + 168시간(주간) 주기 (E=4 if sincos)

    Returns:
        cb(start_idx, H, device) -> Tensor shape [H, E]
    """
    date_type = date_type.upper()

    # 주기 설정 (List of periods)
    # 딥러닝 모델은 절대적인 날짜보다 '반복되는 주기성'을 학습하는 것이 중요합니다.
    if date_type == 'M':
        periods = [12.0]
    elif date_type == 'W':
        periods = [52.0]
    elif date_type == 'D':
        # Day of Week (7), Day of Year (365.25)
        periods = [7.0, 365.25]
    elif date_type == 'H':
        # Hour of Day (24), Hour of Week (24*7=168)
        periods = [24.0, 168.0]
    else:
        # 기본값 (주간 가정)
        periods = [52.0]

    def cb(start_idx: int, H: int, device='cuda' if torch.cuda.is_available() else 'mps'):
        # t: [start_idx, ..., start_idx + H - 1]
        t = torch.arange(start_idx, start_idx + H, device=device, dtype=torch.float32)

        feats = []
        if sincos:
            for p in periods:
                # 각 주기별 sin, cos 쌍 추가 -> (H, 2)
                feats.append(calendar_sin_cos(t, p))

            # (H, 2 * len(periods))
            exo = torch.cat(feats, dim=-1)
        else:
            # Normalized linear ramp [0, 1) for each period
            for p in periods:
                # (t % p) / p -> (H, 1)
                feat = ((t % p) / p).unsqueeze(-1)
                feats.append(feat)
            exo = torch.cat(feats, dim=-1)

        return exo  # [H, E]

    return cb


# ===== 공용 유틸 =====
@torch.no_grad()
def apply_exo_shift_linear(head: nn.Module,
                           future_exo: torch.Tensor,  # (B,H,E) or (H,E)
                           *,
                           horizon: int,
                           out_dtype=None,
                           out_device=None) -> torch.Tensor:  # -> (B,H)
    # 1) head/device/dtype 결정
    if out_device is None:
        try:
            out_device = next(head.parameters()).device
        except StopIteration:
            out_device = future_exo.device
    if out_dtype is None:
        out_dtype = future_exo.dtype

    # 2) 배치 차원 보정
    ex = future_exo
    if ex.dim() == 2:  # (H,E) -> (1,H,E)
        ex = ex.unsqueeze(0)

    # 3) 디바이스/타입 정렬 + head 이동
    ex = ex.to(device=out_device, dtype=out_dtype, non_blocking=True)
    if isinstance(head, nn.Module):
        head = head.to(out_device)

    # 4) 선형 head 적용
    ex = head(ex).squeeze(-1)  # (B,H)

    # 5) H 길이 정합 (pad/trim)
    B, Hx = ex.shape[0], ex.shape[1]
    if Hx < horizon:
        pad = torch.full((B, horizon - Hx), 0.0, device=ex.device, dtype=ex.dtype)
        ex = torch.cat([ex, pad], dim=1)
    elif Hx > horizon:
        ex = ex[:, :horizon]
    return ex


def compose_exo_calendar_age_warranty_cb(
        *,
        date_type: Literal['W', 'M'] = 'W',
        use_sincos: bool = True,
        use_age: bool = True,
        use_warranty: bool = True,
        wty_month: Optional[float] = None,
        age_origin_idx: Optional[int] = None,
        age_norm_mode: Literal['H', 'const', 'none'] = 'H',
        age_norm_div: Optional[float] = None,
        include_in_warranty_flag: bool = True,
        include_time_to_warranty_end: bool = True,
) -> callable:
    """
    (기존 유지) Warranty 관련 로직은 주간/월간 위주로 설계됨.
    Daily/Hourly 지원이 필요하다면 별도 확장이 필요하지만,
    현재 요청 범위(캘린더 주기성)에는 영향을 주지 않으므로 기존 로직을 유지합니다.
    """
    if date_type == 'W':
        period = 52

        def _wty_units(months: float) -> float:
            return float(months) * 4.345
    elif date_type == 'M':
        period = 12

        def _wty_units(months: float) -> float:
            return float(months)
    else:
        # D/H 등 미지원 타입이 들어오면 에러 방지를 위해 기본 W 처리하거나 에러 발생
        # 여기서는 안전하게 W로 폴백하지 않고 에러 유지
        raise ValueError("compose_exo_calendar_age_warranty_cb currently supports only 'W' or 'M'.")

    def _normalize_age(age: torch.Tensor, H: int) -> torch.Tensor:
        if age_norm_mode == 'H':
            denom = float(max(1, H))
            return age / denom
        elif age_norm_mode == 'const':
            denom = float(age_norm_div) if (age_norm_div is not None) else 100.0
            return age / max(1.0, denom)
        else:
            return age

    def cb(start_idx: int, H: int, device='cuda' if torch.cuda.is_available() else 'mps') -> torch.Tensor:
        t = torch.arange(start_idx, start_idx + H, device=device, dtype=torch.float32)
        feats = []

        # 1) sin/cos
        if use_sincos:
            feats.append(torch.sin(2 * torch.pi * t / period).unsqueeze(-1))
            feats.append(torch.cos(2 * torch.pi * t / period).unsqueeze(-1))

        # 2) age (sequence)
        if use_age:
            if age_origin_idx is None:
                age = t
            else:
                age = t - float(age_origin_idx)
                age = torch.clamp(age, min=0.0)
            age = _normalize_age(age, H).unsqueeze(-1)
            feats.append(age)

        # 3) warranty
        if use_warranty and (wty_month is not None):
            w_units = _wty_units(wty_month)
            if age_origin_idx is None:
                age_raw = t
            else:
                age_raw = torch.clamp(t - float(age_origin_idx), min=0.0)

            if include_in_warranty_flag:
                in_wty = (age_raw < w_units).to(torch.float32).unsqueeze(-1)
                feats.append(in_wty)

            if include_time_to_warranty_end:
                rem = torch.clamp(w_units - age_raw, min=0.0)
                rem_norm = (rem / max(1.0, float(w_units))).unsqueeze(-1)
                feats.append(rem_norm)

        if not feats:
            return torch.zeros(H, 0, device=device, dtype=torch.float32)
        return torch.cat(feats, dim=-1)

    return cb

