import math
import torch
import torch.nn as nn

class DecompQuantileHead(nn.Module):
    """
    입력 z (B, D) -> Trend + Seasonal + Irregular 기법 분해 -> 중앙(P50) -> P10/P50/P90 산출
    - 시즌 구성: sin/cos 기저 (n_harmonics)
    - 단조성 보장: softplus delta로 q10<=q50<=q90 유지
    """

    def __init__(self,
                 in_dim: int,
                 horizon: int,
                 quantiles=(0.1, 0.5, 0.9),
                 n_harmonics: int = 4
                 ):
        super().__init__()
        self.h = horizon
        self.qs = tuple(sorted(quantiles))
        assert self.qs == (0.1, 0.5, 0.9)
        self.nh = n_harmonics

        # Trend / Irregular
        self.trend = nn.Linear(in_dim, self.h)
        self.irregular = nn.Sequential(
            nn.Linear(in_dim, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h),
        )

        # Seasonal: B (H, K), coeff (B, K)
        K = 2 * self.nh
        self.season_coeff = nn.Linear(in_dim, K)

        idx = torch.arange(self.h).float() # 0..H-1
        basis = []
        for k in range(1, self.nh + 1):
            basis.append(torch.sin(2 * math.pi * k * idx / max(1, self.h)))
            basis.append(torch.cos(2 * math.pi * k * idx / max(1, self.h)))
        self.register_buffer('basis', torch.stack(basis, dim = 1)) # (H, K)

        # Quantile deltas
        self.delta_low = nn.Linear(in_dim, self.h)
        self.delta_high = nn.Linear(in_dim, self.h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D)
        return: (B, 3, H) for (q10, q50, q90)
        """

        B = z.size(0)
        trend = self.trend(z)               # (B, H)
        coeff = self.season_coeff(z)        # (B, K)
        seasonal = coeff @ self.basis.T     # (B, H)
        irreg = self.irregular(z)           # (B, H)

        center = trend + seasonal + irreg   # q50

        low = center - torch.nn.functional.softplus(self.delta_low(z))
        high = center + torch.nn.functional.softplus(self.delta_high(z))

        q10 = low
        q50 = center
        q90 = high
        return torch.stack([q10, q50, q90], dim = 1) # (B, 3, H)
