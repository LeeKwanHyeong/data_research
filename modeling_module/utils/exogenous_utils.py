import torch
import torch.nn as nn

def calendar_cb(t0: int, H: int, device="cpu"):
    t = torch.arange(t0, t0+H, device=device, dtype=torch.float32)
    return torch.stack([torch.sin(2*torch.pi*t/12), torch.cos(2*torch.pi*t/12)], dim=-1)  # (H,2)


# ===== 공용 유틸 =====
def _apply_exo_shift_linear(head: nn.Module,
                            future_exo: torch.Tensor,
                            horizon: int,
                            out_dtype: torch.dtype,
                            out_device: torch.device) -> torch.Tensor:
    """
    future_exo: (B, Hx, exo_dim) -> head -> (B, Hx, 1) -> squeeze -> (B, Hx)
    Hx != horizon 이면 자동 pad/trim해서 (B, H)로 맞춰 반환.
    """
    if future_exo is None:
        return None

    ex = future_exo.to(device=out_device, dtype=out_dtype, non_blocking=True)
    B, Hx, _ = ex.shape

    ex = head(ex).squeeze(-1)            # (B, Hx, 1) -> (B, Hx)

    if Hx == horizon:
        return ex
    elif Hx > horizon:
        return ex[:, :horizon]
    else:
        pad = torch.zeros(B, horizon - Hx, device=out_device, dtype=out_dtype)
        return torch.cat([ex, pad], dim=1)  # (B, H)