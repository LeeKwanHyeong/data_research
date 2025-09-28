import torch

def calendar_cb(t0: int, H: int, device="cpu"):
    t = torch.arange(t0, t0+H, device=device, dtype=torch.float32)
    return torch.stack([torch.sin(2*torch.pi*t/12), torch.cos(2*torch.pi*t/12)], dim=-1)  # (H,2)
