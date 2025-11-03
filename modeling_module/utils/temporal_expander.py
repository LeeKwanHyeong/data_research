import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalExpander(nn.Module):
    """
    [B, D] -> [B, H, F]
    - 시간 변동성을 강제하기 위해 per-time learnable bias를 직접 가산
    - Fourier PE는 안정적으로(fp32) 계산 후 dtype 복귀
    - 모든 Residual에는 학습 가능한 스케일(α)을 도입해 폭주/NaN 방지
    - 월간/주간 등의 실제 주기를 season_period로 지정(기본 52=주간)
    - Lazy 모듈/forward 내 동적 레이어 생성 없음 (state_dict 안정)

    Args:
        d_in:  입력 임베딩 차원 D
        horizon: 예측 길이 H
        f_out: 시간축 출력 임베딩 차원 F
        dropout: proj 드롭아웃
        use_sinus: Fourier PE 사용 여부
        season_period: Fourier 주기(예: 월간=12, 주간=52)
        max_harmonics: Fourier 사용 최대 고조파 수(과도한 주파수 억제)
        use_conv: 시간 곡률 강제를 위한 depthwise separable conv 사용 여부
    """
    def __init__(
        self,
        d_in: int,
        horizon: int,
        f_out: int = 256,
        dropout: float = 0.05,
        use_sinus: bool = True,
        season_period: int = 52,
        max_harmonics: int = 16,
        use_conv: bool = True,
    ):
        super().__init__()
        self.h = int(horizon)
        self.f = int(f_out)
        self.use_conv = bool(use_conv)
        self.season_period = max(1, int(season_period))
        self.max_harmonics = max(1, int(max_harmonics))
        self.ln = nn.LayerNorm(self.f, eps=1e-5)
        if not hasattr(self, "ln"):
            self.ln = nn.LayerNorm(self.f, eps=1e-5)

        # 1) [B, D] -> [B, F]
        self.proj = nn.Sequential(
            nn.Linear(d_in, self.f),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2) per-time learnable bias (시간 변동성 강제)
        self.time_bias = nn.Parameter(torch.zeros(self.h, self.f))  # [H,F]
        nn.init.normal_(self.time_bias, mean=0.0, std=0.05)

        # 3) Fourier positional encoding (buffer + linear proj)
        self.use_sinus = bool(use_sinus)
        if self.use_sinus:
            # 주기 = season_period (12/52 권장), 고조파 수는 F/2와 max_harmonics 중 작은 값
            K = min(self.f // 2, self.max_harmonics)
            t = torch.arange(self.h, dtype=torch.float32)  # [H]
            feats = []
            for k in range(1, K + 1):
                ang = 2.0 * math.pi * k * (t % self.season_period) / self.season_period
                feats += [torch.sin(ang), torch.cos(ang)]
            # F에 맞춰 zero-pad
            pe = torch.stack(feats, dim=-1) if feats else torch.zeros(self.h, 0, dtype=torch.float32)
            if pe.shape[-1] < self.f:
                pad = self.f - pe.shape[-1]
                pe = torch.cat([pe, torch.zeros(self.h, pad, dtype=torch.float32)], dim=-1)
            elif pe.shape[-1] > self.f:
                pe = pe[:, :self.f]
            self.register_buffer("time_pe", pe, persistent=False)  # [H,F]
            self.time_proj = nn.Linear(self.f, self.f, bias=False)
            nn.init.xavier_uniform_(self.time_proj.weight)
        else:
            self.register_buffer("time_pe", None)

        # 4) γ(PE scale) with bounds, α scales (residuals)
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))  # γ = clamp(0.5 + softplus(raw), [0.1, 2.0])
        self.alpha_pe   = nn.Parameter(torch.tensor(1.6))  # base + α_pe * pe
        self.alpha_bias = nn.Parameter(torch.tensor(1.0))  # base + α_bias * time_bias
        self.alpha_mix  = nn.Parameter(torch.tensor(0.3))  # residual after MLP mixer
        self.alpha_conv = nn.Parameter(torch.tensor(0.3))  # residual after conv

        # 5) Mixer on concat([y0, pe]) -> [B,H,F]
        self.mixer = nn.Sequential(
            nn.Linear(2 * self.f, self.f),
            nn.GELU(),
            nn.Linear(self.f, self.f),
        )
        self.ln1 = nn.LayerNorm(self.f, eps=1e-5)

        # 6) Depthwise Separable Conv for temporal curvature (optional)
        if self.use_conv:
            self.dwconv = nn.Conv1d(self.f, self.f, kernel_size=3, padding=1, groups=self.f)
            self.pwconv = nn.Conv1d(self.f, self.f, kernel_size=1)
            # 보수적 초기화 (폭주 방지)
            nn.init.kaiming_normal_(self.dwconv.weight, nonlinearity='linear')
            with torch.no_grad():
                self.dwconv.weight.mul_(0.1)
                if self.dwconv.bias is not None:
                    self.dwconv.bias.zero_()
            nn.init.kaiming_normal_(self.pwconv.weight, nonlinearity='linear')
            with torch.no_grad():
                self.pwconv.weight.mul_(0.1)
                if self.pwconv.bias is not None:
                    self.pwconv.bias.zero_()
        self.ln2 = nn.LayerNorm(self.f, eps=1e-5)

    # -------- utilities --------
    def _gamma(self) -> torch.Tensor:
        g = 0.5 + F.softplus(self.raw_gamma)
        return torch.clamp(g, 0.1, 2.0)

    def _safe_ln(self, x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        # LayerNorm은 fp32로 계산 후 원 dtype 복귀(AMP 안정)
        out_dtype = x.dtype
        y = ln(x.to(torch.float32)).to(out_dtype)
        return y

    # -------- forward --------
    def forward(self, z_flat: torch.Tensor, step_offset: int = 0) -> torch.Tensor:
        B = z_flat.size(0)
        out_dtype = z_flat.dtype
        offset = int(step_offset) % self.h

        base = self.proj(z_flat).unsqueeze(1).expand(B, self.h, self.f)  # [B,H,F]

        # per-time bias를 offset만큼 회전(phase 연속성)
        bias = self.time_bias.roll(shifts=offset, dims=0)  # [H,F]
        bias = bias.unsqueeze(0).expand(B, -1, -1).to(base.dtype)  # [B,H,F]

        if self.time_pe is not None:
            # 시간 PE도 offset만큼 회전
            pe32 = self.time_proj(self.time_pe.roll(shifts=offset, dims=0))  # [H,F] (fp32)
            pe32 = pe32 / (pe32.norm(dim=-1, keepdim=True) + 1e-4)
            pe = (self._gamma().to(pe32.dtype) * pe32).to(base.dtype)
            pe = pe.unsqueeze(0).expand(B, -1, -1)  # [B,H,F]
        else:
            pe = torch.zeros_like(base)

        y0 = base + self.alpha_pe * pe + self.alpha_bias * bias

        mixed = self.mixer(torch.cat([y0, pe], dim=-1))  # [B,H,F]
        y = y0 + self.alpha_mix * mixed

        # (옵션) conv 잔차 블록이 있다면 기존 코드 유지
        if hasattr(self, "dwconv"):
            yt = y.transpose(1, 2)
            t_feat = self.dwconv(yt)
            t_feat = F.gelu(t_feat)
            t_feat = self.pwconv(t_feat)
            y = y + self.alpha_conv * t_feat.transpose(1, 2)

        y32 = y.to(torch.float32)
        if hasattr(self, "ln") and isinstance(self.ln, nn.LayerNorm):
            y32 = self.ln(y32)
        else:
            # 폴백: 마지막 차원 기준 layer_norm
            y32 = F.layer_norm(y32, (y32.size(-1),), eps=1e-5)
        y = y32.to(out_dtype)
        y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return y
