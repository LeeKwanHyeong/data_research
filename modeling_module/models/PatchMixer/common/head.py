import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecompQuantileHead(nn.Module):
    """
    z (B,D) -> Trend + Seasonal + Irregular -> q50 중심
              softplus 델타로 q10<=q50<=q90
    마지막에 학습형 affine rescale:  y = m + s * q,  s>0
    """

    def __init__(
        self,
        in_dim: int,
        horizon: int,
        quantiles=(0.1, 0.5, 0.9),
        n_harmonics: int = 4,
        # --- rescale 옵션 ---
        add_affine_rescale: bool = True,
        scale_min: float = 1.0,     # softplus 결과에 더해줄 최소 크기(초기 축소 방지)
        global_gain_init: float = 1.0,  # 전체 출력에 곱하는 learnable gain 초기값(선택)
        use_global_gain: bool = False,
    ):
        super().__init__()
        self.h = int(horizon)
        self.qs = tuple(sorted(quantiles))
        assert self.qs == (0.1, 0.5, 0.9), "현재 구현은 (0.1,0.5,0.9) 가정"
        self.nh = int(n_harmonics)

        # Trend / Irregular
        self.trend = nn.Linear(in_dim, self.h)
        self.irregular = nn.Sequential(
            nn.Linear(in_dim, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h),
        )

        # Seasonal basis: (H, K) with K=2*nh
        K = 2 * self.nh
        self.season_coeff = nn.Linear(in_dim, K)
        idx = torch.arange(self.h, dtype=torch.float32)  # 0..H-1
        basis = []
        for k in range(1, self.nh + 1):
            basis.append(torch.sin(2 * math.pi * k * idx / max(1, self.h)))
            basis.append(torch.cos(2 * math.pi * k * idx / max(1, self.h)))
        self.register_buffer("basis", torch.stack(basis, dim=1))  # (H, K)

        # Quantile deltas (softplus로 양수 확보)
        self.delta_low = nn.Linear(in_dim, self.h)
        self.delta_high = nn.Linear(in_dim, self.h)

        # --- 학습형 rescale(시점별) ---
        self.add_affine_rescale = add_affine_rescale
        if self.add_affine_rescale:
            self.shift = nn.Linear(in_dim, self.h)               # m(B,H)
            self.scale_raw = nn.Linear(in_dim, self.h)           # s_raw(B,H) -> softplus -> s>0
            self.scale_min = float(scale_min)

        # --- 전역 gain(선택) ---
        self.use_global_gain = use_global_gain
        if self.use_global_gain:
            self.out_gain = nn.Parameter(torch.tensor(float(global_gain_init)))

        # 간단 초기화(원하면 더 공격적으로)
        nn.init.zeros_(self.delta_low.weight);  nn.init.zeros_(self.delta_low.bias)
        nn.init.zeros_(self.delta_high.weight); nn.init.zeros_(self.delta_high.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D)
        return: (B, 3, H) = (q10, q50, q90)
        """
        B = z.size(0)

        trend = self.trend(z)                 # (B,H)
        coeff = self.season_coeff(z)          # (B,K)
        # basis: (H,K) -> (B,H)
        seasonal = coeff @ self.basis.T
        irreg = self.irregular(z)             # (B,H)

        center = trend + seasonal + irreg     # q50

        # monotonic deltas
        d_low = F.softplus(self.delta_low(z))
        d_high = F.softplus(self.delta_high(z))

        q50 = center
        q10 = q50 - d_low
        q90 = q50 + d_high

        out = torch.stack([q10, q50, q90], dim=1)  # (B,3,H)

        # --- 학습형 rescale ---
        if self.add_affine_rescale:
            m = self.shift(z).unsqueeze(1)                    # (B,1,H)
            s = F.softplus(self.scale_raw(z)).unsqueeze(1)    # (B,1,H) > 0
            if self.scale_min > 0.0:
                s = s + self.scale_min                        # 최소 스케일 보장
            out = m + s * out                                 # (B,3,H)

        if self.use_global_gain:
            out = self.out_gain * out

        return out
