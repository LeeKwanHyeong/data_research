import torch
import torch.nn as nn
from typing import Literal, Optional

def calendar_sin_cos(t0: int, H: int, device="cpu", date_type: Literal['M','W','D']='W'):
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

    def cb(start_idx: int, H: int, device="cpu"):
        t = torch.arange(start_idx, start_idx + H, device=device, dtype=torch.float32)
        if sincos:
            exo = torch.stack([torch.sin(2*torch.pi*t/period),
                               torch.cos(2*torch.pi*t/period)], dim=-1)  # (H,2)
        else:
            exo = (t % period) / period                                    # (H,)
            exo = exo.unsqueeze(-1)                                       # (H,1)
        return exo  # (H,E)
    return cb


@torch.no_grad()
def warranty_features_for_batch(
    start_idx: int,
    H: int,
    *,
    expiry_idx_b: torch.Tensor,
    sigma: float = 2.0,
    norm_k: float = 10.0,
    device="cpu"
) -> torch.Tensor:
    # (기존 그대로)
    B = int(expiry_idx_b.numel())
    t = torch.arange(start_idx, start_idx + H, device=device, dtype=torch.float32)  # [H]
    tB = t.unsqueeze(0).expand(B, -1)                       # [B,H]
    expB = expiry_idx_b.to(device).view(B, 1).float()       # [B,1]
    step = (tB >= expB).float()
    diff = tB - expB
    bump = torch.exp(-0.5 * (diff / (sigma + 1e-6)) ** 2)
    tte  = torch.clamp((expB - tB) / norm_k, min=-1.0, max=1.0)
    feats = torch.stack([step, bump, tte], dim=-1)          # [B,H,3]
    return feats


@torch.no_grad()
def compose_exo_warranty_cb(
    get_expiry_idx_fn,
    *,
    add_calendar: bool = True,
    date_type: Literal['M','W'] = 'W',
    sigma: float = 2.0,
    norm_k: float = 10.0,
):
    """
    반환: future_exo_cb(start_idx, H, device, batch_meta) -> [B,H,D_exo]
    - batch_meta로부터 각 샘플별 만료 절대주차를 받아 warranty-features 생성
    - add_calendar=True면 sin/cos를 앞에 concat ([B,H,2] + [B,H,3] = [B,H,5])
    """
    def _cb(start_idx: int, H: int, device="cpu", batch_meta=None):
        expiry_idx_b = get_expiry_idx_fn(batch_meta)  # [B]
        wty = warranty_features_for_batch(
            start_idx, H, expiry_idx_b=expiry_idx_b,
            sigma=sigma, norm_k=norm_k, device=device
        )  # [B,H,3]
        if add_calendar:
            cal = calendar_sin_cos(start_idx, H, device=device, date_type=date_type)  # [H,2]
            calB = cal.unsqueeze(0).expand(wty.size(0), -1, -1)                      # [B,H,2]
            exo = torch.cat([calB, wty], dim=-1)                                     # [B,H,5]
        else:
            exo = wty
        return exo
    return _cb


# ===== 공용 유틸 =====
def apply_exo_shift_linear(head: nn.Module,
                            future_exo: torch.Tensor,
                            horizon: int,
                            out_dtype: torch.dtype,
                            out_device: torch.device) -> torch.Tensor:
    """
    future_exo: (B, Hx, exo_dim) -> head -> (B, Hx, 1) -> (B, Hx)
    Hx != horizon이면 pad/trim으로 자동 보정 후 (B,H) 반환
    """
    if future_exo is None:
        return None
    ex = future_exo.to(device=out_device, dtype=out_dtype, non_blocking=True)
    B, Hx, _ = ex.shape
    ex = head(ex).squeeze(-1)  # (B, Hx)
    if Hx == horizon:
        return ex
    elif Hx > horizon:
        return ex[:, :horizon]
    else:
        pad = torch.zeros(B, horizon - Hx, device=out_device, dtype=out_dtype)
        return torch.cat([ex, pad], dim=1)
