import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalExpander(nn.Module):
    def __init__(self, d_in, horizon, f_out=128, dropout=0.1,
                 use_sinus=True, season_period=52, max_harmonics=16, use_conv=True):
        super().__init__()
        self.h = horizon
        self.d_in = d_in
        self.use_conv = use_conv

        # 1) 시간 임베딩: 학습 쿼리 + (옵션) Fourier
        self.query = nn.Parameter(torch.randn(horizon, d_in))  # [H, D]
        self.use_sinus = use_sinus
        self.season_period = season_period
        self.max_harmonics = max_harmonics
        if use_sinus:
            freqs = torch.arange(1, max_harmonics + 1).float()
            self.register_buffer("freqs", freqs, persistent=False)
            self.pe_scale = nn.Parameter(torch.tensor(1.0))

        # 2) 시간 bias 분기 (가시적 per-step 변화 강제)
        pe_dim = d_in + (2 * max_harmonics if use_sinus else 0)
        self.time_bias = nn.Sequential(
            nn.Linear(pe_dim, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_in)
        )

        # 3) (옵션) FiLM 변조 (있다면 유지)
        self.film = nn.Sequential(
            nn.Linear(pe_dim, 2 * d_in),
            nn.GELU(),
            nn.Linear(2 * d_in, 2 * d_in)
        )
        self.film_scale = nn.Parameter(torch.tensor(0.5))

        # 4) z vs time_bias 혼합 게이트 (학습 가능, 초기엔 time_bias 쪽 가중↑)
        self.mix_logit = nn.Parameter(torch.tensor(-1.5))  # sigmoid(-1.5)≈0.18 → time_bias 비중 높임

        # 5) 최종 투영 + (옵션) H축 depthwise conv
        self.proj = nn.Sequential(
            nn.Linear(d_in, f_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(f_out, f_out)
        )
        if use_conv:
            self.dw = nn.Conv1d(f_out, f_out, 3, padding=1, groups=f_out)
            self.pw = nn.Conv1d(f_out, f_out, 1)
            self.conv_dropout = nn.Dropout(dropout)

    def _fourier(self, H, device):
        t = torch.arange(H, device=device).float()[:, None]        # [H,1]
        w = 2 * math.pi * t * (self.freqs[None, :] / self.season_period)  # [H,K]
        sin = torch.sin(w); cos = torch.cos(w)
        return torch.cat([sin, cos], dim=-1) * self.pe_scale       # [H,2K]

    def forward(self, z):  # z: [B, D]
        B, D = z.shape
        device = z.device
        Z = z.unsqueeze(1).expand(B, self.h, D)                    # [B,H,D]

        # 시간 임베딩 만들기
        pe = self.query                                           # [H,D]
        if self.use_sinus:
            pe = torch.cat([pe, self._fourier(self.h, device)], dim=-1)  # [H, D+2K]
        pe = pe.unsqueeze(0).expand(B, self.h, -1)                        # [B,H,*]

        # (A) 확실한 per-step bias
        bias = self.time_bias(pe)                                  # [B,H,D]

        # (B) (옵션) FiLM
        gb = self.film(pe)                                         # [B,H,2D]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.sigmoid(gamma)
        beta = torch.tanh(beta) * self.film_scale
        z_film = gamma * Z + beta                                  # [B,H,D]

        # (C) 혼합 (mix_gate 작을수록 time_bias 반영↑)
        mix_gate = torch.sigmoid(self.mix_logit)                   # scalar in (0,1)
        Z_mod = mix_gate * z_film + (1.0 - mix_gate) * (Z + bias)  # ★ 방송 탈출 핵심

        # 투영
        Y = self.proj(Z_mod)                                       # [B,H,F]

        # (옵션) 국소 곡률
        if self.use_conv:
            Yc = self.dw(Y.transpose(1, 2))
            Yc = F.gelu(Yc)
            Yc = self.pw(Yc)
            Yc = self.conv_dropout(Yc)
            Y = Y + Yc.transpose(1, 2)
        return Y