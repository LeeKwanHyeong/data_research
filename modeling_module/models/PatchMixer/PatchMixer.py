import torch
import torch.nn as nn

from modeling_module.models.PatchMixer.backbone import PatchMixerBackbone, MultiScalePatchMixerBackbone
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.common_layers.RevIN import RevIN
from modeling_module.models.common_layers.heads.quantile_heads.decomposition_quantile_head import \
    DecompositionQuantileHead
from modeling_module.utils.exogenous_utils import apply_exo_shift_linear
from modeling_module.utils.temporal_expander import TemporalExpander

# -------------------------
# Simple PatchMixer -> Horizon regression
# -------------------------
class BaseModel(nn.Module):
    """
    PatchMixer Backbone → TemporalExpander → per-step head
    + base(절편+기울기, α-게이트) + step-gate(Conv1d+τ) + DW residual
    """
    def __init__(self, configs):
        super().__init__()
        self.model_name = 'PatchMixer BaseModel'

        self.horizon = configs.horizon
        self.f_out = configs.expander_f_out



        self.backbone = PatchMixerBackbone(configs=configs)
        in_dim = self.backbone.patch_repr_dim

        self.expander = TemporalExpander(
            d_in = in_dim, horizon = self.horizon, f_out = self.f_out, dropout = 0.1,
            use_sinus = True,
            season_period = int(getattr(configs, 'expander_season_period', 52)),
            max_harmonics = int(getattr(configs, 'expander_max_harmonics', 16)),
            use_conv = True
        )

        # RevIN(norm 전용; denorm은 forecaster)
        self.revin = RevIN(configs.enc_in)

        # base(절편 + 기울기) + base gate α
        self.base_head_b = nn.Linear(in_dim, 1)
        self.base_head_m = nn.Linear(in_dim, 1)
        self.base_gate   = nn.Linear(in_dim, 1)
        nn.init.constant_(self.base_gate.bias, -2.5)  # 초기엔 resid 쪽이 크게

        self.pre_ln = nn.LayerNorm(self.f_out)
        self.head = nn.Sequential(
            nn.Linear(self.f_out, self.f_out),
            nn.GELU(),
            nn.Linear(self.f_out, 1)
        )

        self.resid_scale = nn.Parameter(torch.tensor(1.2))

        # ---- Step gate: H-방향 Conv + τ 가법 ----
        self.gate_ln = nn.LayerNorm(self.f_out)


        # 멀티스케일 컨볼루션: 3x, 5x, dilated-3 병렬 후 1x1로 축소
        self.gate_conv_3 = nn.Conv1d(self.f_out, 32, kernel_size=3, padding=1, dilation=1)
        self.gate_conv_5 = nn.Conv1d(self.f_out, 32, kernel_size=5, padding=2, dilation=1)
        self.gate_conv_d3 = nn.Conv1d(self.f_out, 32, kernel_size=3, padding=2, dilation=2)
        self.gate_reduce = nn.Conv1d(96, 1, kernel_size=1)  # 32*3 -> 1
        self.gate_act = nn.GELU()
        self.gate_do = nn.Dropout(0.1)

        # τ 영향도/게인/바이어스/온도/클램프
        self.tau_weight = nn.Parameter(torch.tensor(1.0))  # 0.5 -> 1.0
        self.g_gain = nn.Parameter(torch.tensor(5.0))  # 로짓 스케일↑
        self.g_bias = nn.Parameter(torch.tensor(1.8))  # 로짓 바이어스
        self.gate_temp = nn.Parameter(torch.tensor(1.0))  # 1.5 -> 1.0 (감도↑)
        self.g_logit_clip = 8.0

        # 출력 스케일/바이어스
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.out_bias  = nn.Parameter(torch.tensor(0.0))

        # H축 depthwise residual(국소 곡률)
        self.dw_head = nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1)
        self.dw_gain = nn.Parameter(torch.tensor(1.0))

        # 외생
        self.exo_dim = int(configs.exo_dim)
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )
        else:
            self.exo_head = None

        self.final_nonneg = True  # 추론시에만 clamp

    def forward(self,
                x: torch.Tensor,
                future_exo: torch.Tensor | None = None,
                *,
                exo_is_normalized: bool = True
                ) -> torch.Tensor:
        # 1) 정규화(denorm은 forecaster)
        x = self.revin(x, 'norm')                 # [B,L,C]
        z = self.backbone(x)                      # [B,D]
        x_bhf = self.expander(z)                  # [B,H,F]
        x_bhf_n = self.pre_ln(x_bhf)              # [B,H,F]

        B, H = z.size(0), self.horizon
        # t01 = torch.linspace(0, 1, H, device=z.device).unsqueeze(0)
        # t = t01 * 0.7  # 기울기 과대 억제
        t = torch.linspace(-1, 1, H, device=z.device).unsqueeze(0)

        # 2) base + α
        b = self.base_head_b(z)                   # [B,1]
        m = self.base_head_m(z)                   # [B,1]
        base = b + m * t                          # [B,H]
        alpha = torch.sigmoid(self.base_gate(z)).expand(-1, H)  # [B,H]

        # 3) residual
        resid = self.head(x_bhf_n).squeeze(-1)    # [B,H]
        resid = self.resid_scale * resid
        resid = resid - resid.mean(dim=1, keepdim=True)  # 잔차 평균 0


        # 4) step gate (Conv1d on H + τ)
        xg = self.gate_ln(x_bhf_n).transpose(1, 2)  # [B,F,H]
        g1 = self.gate_act(self.gate_conv_3(xg))  # [B,32,H]
        g2 = self.gate_act(self.gate_conv_5(xg))  # [B,32,H]
        g3 = self.gate_act(self.gate_conv_d3(xg))  # [B,32,H]
        gcat = torch.cat([g1, g2, g3], dim=1)  # [B,96,H]
        gcat = self.gate_do(gcat)
        g_logit = self.gate_reduce(gcat).transpose(1, 2).squeeze(-1)  # [B,H]

        tau = torch.linspace(-1.0, 1.0, H, device=x_bhf.device).view(1, H).expand(B, H)
        g_logit = (g_logit + self.tau_weight * tau + self.g_bias)
        g_logit = torch.clamp(self.g_gain * (g_logit / self.gate_temp), -self.g_logit_clip, self.g_logit_clip)
        gate = torch.sigmoid(g_logit)  # [B,H]
        gate = gate - gate.mean(dim=1, keepdim=True) + 0.5
        gate = torch.clamp(gate, 0.05, 0.95)  # 과포화 방지


        # 5) 혼합
        y = alpha * base + (1.0 - alpha) * (gate * resid)          # [B,H]

        # 6) exogenous(정규화 공간 기준이면 여기서 더함)
        if (self.exo_head is not None) and (future_exo is not None):
            ex = apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon = self.horizon,
                out_dtype = y.dtype,
                out_device = y.device
            )  # (B, H)
            if exo_is_normalized:
                # RevIN 기준 normalize space에서 더하고, 이후 한 번에 denorm
                y = y + ex
            else:
                # 원단위 exo면 denorm 이후에 가산
                pass

        # 7) scale/bias + H축 DW 곡률
        y = y * self.out_scale + self.out_bias
        yc = self.dw_head(y.unsqueeze(1)).squeeze(1)
        y  = y + self.dw_gain * yc

        y = self.revin(y.unsqueeze(-1), 'denorm').squeeze(-1)
        if (self.exo_head is not None) and (future_exo is not None) and (not exo_is_normalized):
            # 원단위 exo는 역정규화 이후에 가산
            y = y + ex

        # 8) 추론시에만 비음수 클램프
        if getattr(self, 'final_nonneg', False) and (not self.training):
            y = torch.clamp_min(y, 0.0)


        return y

# -------------------------
# Simple PatchMixer + Decomposition Quantile Head
# -------------------------
class QuantileModel(nn.Module):
    """
    Multi-Scale PatchMixer Backbone + DecompQuantileHeadV2
    output: (B, 3, H)  # (q10, q50, q90)
    + (선택) exogenous shift 동일 적용
    """
    def __init__(self,
                 configs: PatchMixerConfig,
                 ):
        super().__init__()
        self.is_quantile = True
        self.model_name = 'PatchMixer QuantileModel'
        self.patch_cfgs = configs.patch_cfgs
        self.fused_dim = configs.fused_dim
        self.horizon = configs.horizon
        self.per_branch_dim = configs.per_branch_dim
        self.fusion = configs.fusion
        self.n_harmonics = configs.expander_n_harmonics
        self.exo_dim = configs.exo_dim
        self.f_out = configs.expander_f_out


        # 1) Backbone: 전역 벡터 [B, D]
        self.backbone = MultiScalePatchMixerBackbone(
            base_configs=configs,
            patch_cfgs=self.patch_cfgs,
            per_branch_dim=self.per_branch_dim,
            fused_dim=self.fused_dim,
            fusion=self.fusion,
        )
        d_in = self.backbone.out_dim

        # 2) Temporal Expander: [B,D] -> [B,H,F]
        self.expander = TemporalExpander(
            d_in=d_in, horizon=self.horizon, f_out=self.f_out, dropout=0.1,
            use_sinus=True,
            season_period=int(getattr(configs, "season_period", 52)),
            max_harmonics=int(getattr(configs, "max_harmonics", 16)),
            use_conv=True
        )

        # 3) Decomposition Quantile Head (V2): [B,H,F] -> [B,Q,H]
        self.head = DecompositionQuantileHead(
            in_features=self.f_out,
            quantiles=[0.1, 0.5, 0.9],
            hidden=128,
            dropout=float(getattr(configs, 'head_dropout', 0.0) or 0.0),
            mid=0.5,
            use_trend=True,
            fourier_k=self.n_harmonics,
            agg="mean",
        )

        # (선택) exogenous shift
        self.exo_dim = int(self.exo_dim)
        self.exo_head = None
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

        self.revin = RevIN(configs.enc_in)  # enc_in=1 가정

    def _denorm_quantiles_with_revin(self, q_bqh: torch.Tensor) -> torch.Tensor:
        """
        q_bqh: [B,Q,H]  (Q=3)
        RevIN은 [B,L,N] 형태(또는 [B,L] 변형)에 맞춰 denorm해야 하므로,
        분위수별로 [B,H,1]로 만들어 각각 denorm 적용 후 다시 합칩니다.
        """
        B, Q, H = q_bqh.shape
        outs = []
        for i in range(Q):
            yi = q_bqh[:, i, :]            # [B,H]
            yi = yi.unsqueeze(-1)          # [B,H,1]  (N=1)
            yi = self.revin(yi, 'denorm')  # RevIN이 [B,L,N]을 받는다고 가정
            outs.append(yi.squeeze(-1))    # [B,H]
        return torch.stack(outs, dim=1)     # [B,Q,H]

    def _ensure_bqh(self, q: torch.Tensor, horizon: int, qlen: int) -> torch.Tensor:
        # 허용: (B,Q,H) 또는 (B,H,Q)
        if q.dim() != 3:
            raise ValueError(f"pred must be 3D, got {q.shape}")
        B, A, Bdim = q.shape
        if A == qlen and Bdim == horizon:  # (B,Q,H)
            return q
        if A == horizon and Bdim == qlen:  # (B,H,Q)
            return q.permute(0, 2, 1).contiguous()
        raise ValueError(f"pred shape must be (B,{qlen},{horizon}) or (B,{horizon},{qlen}), got {q.shape}")

    def forward(self, x: torch.Tensor, future_exo: torch.Tensor | None = None, *,
                exo_is_normalized: bool = True, **kwargs):
        """
        x: (B, L, N)  # RevIN이 이 형태를 받는 구현 가정
        return: (B, 3, H)
        """
        # 0) 입력 정규화
        x_n = self.revin(x, 'norm')

        # 1) 백본: 전역 벡터
        z = self.backbone(x_n)               # (B, D)

        # 2) 시점 확장
        x_bhf = self.expander(z)             # (B, H, F)

        # 3) 분위수 예측(교차 방지 포함)
        q = self.head(x_bhf)                 # (B, 3, H)  normalized space

        q = self._ensure_bqh(q, self.horizon, qlen=3)

        # 4) (선택) exogenous shift
        #    - exo_is_normalized=True: RevIN 기준 공간에서 학습/입력된 exo라면 denorm 이전에 더함
        #    - exo_is_normalized=False: 원 단위라면 denorm 이후에 더해야 함
        if (self.exo_head is not None) and (future_exo is not None) and exo_is_normalized:
            ex = apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon=self.horizon,
                out_dtype=q.dtype,
                out_device=q.device
            )  # (B, H)
            q = q + ex.unsqueeze(1)          # (B, 3, H)


        # 6) exogenous가 원 단위일 때는 여기서 더하세요.
        if (self.exo_head is not None) and (future_exo is not None) and (not exo_is_normalized):
            ex = apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon=self.horizon,
                out_dtype=q.dtype,
                out_device=q.device
            )  # (B, H)
            q = q + ex.unsqueeze(1)          # (B, 3, H)

        qs = []
        for i in range(q.size(-1)):
            qi_raw = self.revin(q[..., i: i+1], 'denorm')
            qs.append(qi_raw)
        q_raw = torch.cat(qs, dim = -1)

        if getattr(self, 'final_nonneg', False) and (not self.training):
            q_raw = torch.clamp_min(q_raw, 0.0)

        # # 5) RevIN 역정규화 (분위수별 슬라이스)
        # q = self._denorm_quantiles_with_revin(q)  # (B, 3, H)

        return {"q": q_raw}

