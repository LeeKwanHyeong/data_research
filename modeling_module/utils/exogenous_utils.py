import torch
import torch.nn as nn
from typing import Literal, Optional

def calendar_sin_cos(t0: int, H: int, device='cuda' if torch.cuda.is_available() else 'mps', date_type: Literal['M','W','D']='W'):
    if date_type == 'M':
        period = 12
    elif date_type == 'W':
        period = 52
    else:  # 'D'
        period = 24
    t = torch.arange(t0, t0 + H, device=device, dtype=torch.float32)
    return torch.stack(
        [torch.sin(2 * torch.pi * t / period), torch.cos(2 * torch.pi * t / period)],
        dim=-1
    )  # (H,2)


def compose_exo_calendar_cb(date_type: str = "W", *, sincos: bool = True):
    period = 52 if date_type.upper().startswith("W") else 12
    E = 2 if sincos else 1

    def cb(start_idx: int, H: int, device='cuda' if torch.cuda.is_available() else 'mps'):
        t = torch.arange(start_idx, start_idx + H, device=device, dtype=torch.float32)
        if sincos:
            exo = torch.stack([torch.sin(2*torch.pi*t/period),
                               torch.cos(2*torch.pi*t/period)], dim=-1)  # (H,2)
        else:
            exo = (t % period) / period                                    # (H,)
            exo = exo.unsqueeze(-1)                                       # (H,1)
        return exo  # (H,E)
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



'''사용 예시
# 주차(YYYYWW) 기준, sin/cos + age/H + in_warranty + 남은기간(정규화)
future_exo_cb = compose_exo_calendar_age_warranty_cb(
    date_type='W',
    use_sincos=True,
    use_age=True,
    use_warranty=True,
    wty_month=24.0,          # 파트별로 다르면 파트 루프에서 주입
    age_origin_idx=first_idx, # 해당 파트의 최초 판매 절대 index
    age_norm_mode='H',        # age/H
)

# 추론 시: dataset/collate에서
fe = future_exo_cb(start_idx, H, device='cpu')  # (H, E)'''


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
    Returns:
        cb(start_idx: int, H: int, device='cpu') -> Tensor[H, E]

    Features (in order, if enabled):
      1) sin, cos (period = 52 if 'W', 12 if 'M')
      2) age (절대/상대 시퀀스; 정규화 옵션)
      3) warranty:
          - in_warranty (0/1)
          - time_to_warranty_end (0~1 정규화)

    Args:
      date_type: 'W'(주차) 또는 'M'(월차)
      use_sincos: 캘린더 계절성 사용 여부
      use_age: 절대 순서(age) 사용 여부
      use_warranty: 워런티 관련 피처 사용 여부
      wty_month: 보증 개월 (None이면 워런티 피처 미생성)
      age_origin_idx: age를 0으로 두고 싶은 기준 인덱스(절대 index). None이면 t 자체를 age로 사용
      age_norm_mode:
         - 'H'    : age / H
         - 'const': age / (age_norm_div 또는 100.0)
         - 'none' : 정규화 없음
      include_in_warranty_flag: in_warranty(0/1) 포함 여부
      include_time_to_warranty_end: 보증 종료까지 남은 기간(0~1) 포함 여부
    """
    if date_type == 'W':
        period = 52
        # 'W'일 때 보증 기간 단위는 '주'로 환산
        def _wty_units(months: float) -> float:
            return float(months) * 4.345  # 월→주 근사
    elif date_type == 'M':
        period = 12
        # 'M'일 때 보증 기간 단위는 '월'
        def _wty_units(months: float) -> float:
            return float(months)
    else:
        raise ValueError("date_type must be 'W' or 'M'.")

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
        # 절대 인덱스 t: [start_idx, ..., start_idx+H-1]
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
                # 음수 방지(옵션): 보통 추론 시에는 start_idx >= age_origin_idx라 0 이상이나,
                # 안전하게 음수면 0으로 클리핑
                age = torch.clamp(age, min=0.0)
            age = _normalize_age(age, H).unsqueeze(-1)
            feats.append(age)

        # 3) warranty
        if use_warranty and (wty_month is not None):
            w_units = _wty_units(wty_month)  # 주/월 단위로 환산된 보증 기간
            # age가 없는 경우를 대비하여 age_raw 정의
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
        return torch.cat(feats, dim=-1)  # (H, E)

    return cb
