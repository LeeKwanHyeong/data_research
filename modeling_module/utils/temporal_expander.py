import torch
import torch.nn as nn

class TemporalExpander(nn.Module):
    """
    z_flat: [B, D] -> x_bhf: [B, H, F]
    간단한 horizon 임베딩(pe_t)과 z를 결합해 시점별 분화를 만듭니다.
    """
    def __init__(self, d_in: int, horizon: int, f_out: int = 128, dropout: float = 0.1):
        super().__init__()
        self.H = int(horizon)
        self.pe = nn.Parameter(torch.randn(self.H, f_out))  # [H, F]
        self.proj = nn.Sequential(
            nn.Linear(d_in + f_out, f_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        # z_flat: [B, D]
        B, D = z_flat.shape
        z_rep = z_flat.unsqueeze(1).expand(B, self.H, D)   # [B,H,D]
        pe_B  = self.pe.unsqueeze(0).expand(B, -1, -1)     # [B,H,F]
        x     = torch.cat([z_rep, pe_B], dim=-1)           # [B,H,D+F]
        return self.proj(x)                                 # [B,H,F]