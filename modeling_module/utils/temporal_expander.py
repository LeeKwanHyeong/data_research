import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalExpander(nn.Module):
    """
    [B, D_in] → (시간조건) → [B, H, F]
    - 학습가능 쿼리 + (옵션) Fourier 기반 시간임베딩
    - 시간임베딩으로 z를 FiLM(gamma,beta) 변조
    - time_bias를 직접 더해 per-step 차이 강제
    - (옵션) H축 depthwise conv로 국소 곡률 부여
    """
    def __init__(self, d_in: int, horizon: int, f_out: int = 128, dropout: float = 0.1,
                 use_sinus: bool = True, season_period: int = 52, max_harmonics: int = 16,
                 use_conv: bool = True):
        super().__init__()
        self.h = int(horizon)
        self.d_in = int(d_in)
        self.use_conv = bool(use_conv)
        self.use_sinus = bool(use_sinus)
        self.season_period = int(season_period)
        self.max_harmonics = int(max_harmonics)

        # 시간 쿼리
        self.query = nn.Parameter(torch.randn(self.h, self.d_in))  # [H, D_in]

        # Fourier
        if self.use_sinus:
            freqs = torch.arange(1, self.max_harmonics + 1).float()
            self.register_buffer("freqs", freqs, persistent=False)
            self.pe_scale = nn.Parameter(torch.tensor(1.0))

        pe_dim = self.d_in + (2 * self.max_harmonics if self.use_sinus else 0)

        # per-step bias
        self.time_bias = nn.Sequential(
            nn.Linear(pe_dim, self.d_in),
            nn.GELU(),
            nn.Linear(self.d_in, self.d_in)
        )

        # FiLM
        self.film = nn.Sequential(
            nn.Linear(pe_dim, 2 * self.d_in),
            nn.GELU(),
            nn.Linear(2 * self.d_in, 2 * self.d_in)
        )
        self.film_scale = nn.Parameter(torch.tensor(0.5))

        # z vs time_bias 혼합 게이트 (처음엔 bias 비중을 조금 더)
        self.mix_logit = nn.Parameter(torch.tensor(-1.5))  # sigmoid≈0.18

        # 최종 투영
        self.proj = nn.Sequential(
            nn.Linear(self.d_in, f_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(f_out, f_out)
        )

        if self.use_conv:
            self.dw = nn.Conv1d(f_out, f_out, kernel_size=3, padding=1, groups=f_out)
            self.pw = nn.Conv1d(f_out, f_out, kernel_size=1)
            self.conv_dropout = nn.Dropout(dropout)

    def _fourier(self, H: int, device):
        t = torch.arange(H, device=device).float()[:, None]  # [H,1]
        w = 2 * math.pi * t * (self.freqs[None, :] / self.season_period)  # [H,K]
        sin = torch.sin(w); cos = torch.cos(w)
        return torch.cat([sin, cos], dim=-1) * self.pe_scale  # [H,2K]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D_in]
        B, D = z.shape
        device = z.device

        Z = z.unsqueeze(1).expand(B, self.h, D)  # [B,H,D]

        pe = self.query  # [H,D_in]
        if self.use_sinus:
            pe = torch.cat([pe, self._fourier(self.h, device)], dim=-1)  # [H, D_in+2K]
        pe = pe.unsqueeze(0).expand(B, self.h, -1)  # [B,H,*]

        bias = self.time_bias(pe)  # [B,H,D]

        gb = self.film(pe)         # [B,H,2D]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.sigmoid(gamma)
        beta  = torch.tanh(beta) * self.film_scale
        z_film = gamma * Z + beta  # [B,H,D]

        mix_gate = torch.sigmoid(self.mix_logit)  # scalar
        Z_mod = mix_gate * z_film + (1.0 - mix_gate) * (Z + bias)

        Y = self.proj(Z_mod)  # [B,H,F]

        if self.use_conv:
            Yc = Y.transpose(1, 2)          # [B,F,H]
            Yc = self.dw(Yc)
            Yc = F.gelu(Yc)
            Yc = self.pw(Yc)
            Yc = self.conv_dropout(Yc)
            Y  = Y + Yc.transpose(1, 2)

        return Y
