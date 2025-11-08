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
# PatchMixer -> Horizon regression (Point)
# -------------------------
class BaseModel(nn.Module):
    """
    PatchMixer Backbone → TemporalExpander → per-step head
    + base(절편+기울기, α-게이트) + step-gate(Conv1d+τ) + DW residual
    + (옵션) part embedding, EOL prior, final_nonneg 등
    """
    def __init__(self, configs: PatchMixerConfig):
        super().__init__()
        self.model_name = 'PatchMixer BaseModel'
        self.configs = configs

        self.horizon = configs.horizon
        self.f_out = int(getattr(configs, 'expander_f_out', 128))

        # flags
        self.exo_is_normalized_default = bool(getattr(configs, 'exo_is_normalized_default', True))
        self.final_nonneg = bool(getattr(configs, 'final_nonneg', True))
        self.use_eol_prior = bool(getattr(configs, 'use_eol_prior', False))
        self.eol_feature_index = int(getattr(configs, 'eol_feature_index', 0))

        # Backbone → [B, D]
        self.backbone = PatchMixerBackbone(configs=configs)
        in_dim = self.backbone.patch_repr_dim

        # (옵션) Part Embedding 추가 → z concat 후 차원 복원
        self.use_part_embedding = bool(getattr(configs, 'use_part_embedding', False))
        self.part_emb = None
        self.z_fuser = None
        if self.use_part_embedding and int(getattr(configs, 'part_vocab_size', 0)) > 0:
            pdim = int(getattr(configs, 'part_embed_dim', 16))
            self.part_emb = nn.Embedding(int(configs.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(in_dim + pdim, in_dim)

        # Temporal Expander: [B,D] -> [B,H,F]
        self.expander = TemporalExpander(
            d_in=in_dim, horizon=self.horizon, f_out=self.f_out, dropout=float(getattr(configs, 'dropout', 0.1)),
            use_sinus=True,
            season_period=int(getattr(configs, 'expander_season_period', 52)),
            max_harmonics=int(getattr(configs, 'expander_max_harmonics', 16)),
            use_conv=True
        )

        # RevIN(norm 전용; denorm은 forecaster/모델 내부)
        self.revin = RevIN(int(getattr(configs, 'enc_in', 1)))

        # base(절편 + 기울기) + base gate α
        self.base_head_b = nn.Linear(in_dim, 1)
        self.base_head_m = nn.Linear(in_dim, 1)
        self.base_gate   = nn.Linear(in_dim, 1)
        nn.init.constant_(self.base_gate.bias, -2.5)  # 초기엔 resid 쪽이 크게

        # main residual head
        head_hidden = int(getattr(configs, 'head_hidden', self.f_out))
        self.pre_ln = nn.LayerNorm(self.f_out)
        self.head = nn.Sequential(
            nn.Linear(self.f_out, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1)
        )

        self.resid_scale = nn.Parameter(torch.tensor(1.2))

        # ---- Step gate: H-방향 Conv + τ 가법 ----
        self.gate_ln = nn.LayerNorm(self.f_out)
        self.gate_conv_3 = nn.Conv1d(self.f_out, 32, kernel_size=3, padding=1, dilation=1)
        self.gate_conv_5 = nn.Conv1d(self.f_out, 32, kernel_size=5, padding=2, dilation=1)
        self.gate_conv_d3 = nn.Conv1d(self.f_out, 32, kernel_size=3, padding=2, dilation=2)
        self.gate_reduce = nn.Conv1d(96, 1, kernel_size=1)  # 32*3 -> 1
        self.gate_act = nn.GELU()
        self.gate_do = nn.Dropout(0.1)

        # τ 영향도/게인/바이어스/온도/클램프
        self.tau_weight = nn.Parameter(torch.tensor(1.0))
        self.g_gain = nn.Parameter(torch.tensor(5.0))
        self.g_bias = nn.Parameter(torch.tensor(1.8))
        self.gate_temp = nn.Parameter(torch.tensor(1.0))
        self.g_logit_clip = 8.0

        # 출력 스케일/바이어스
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.out_bias  = nn.Parameter(torch.tensor(0.0))

        # H축 depthwise residual(국소 곡률)
        self.dw_head = nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1)
        self.dw_gain = nn.Parameter(torch.tensor(1.0))

        # 외생
        self.exo_dim = int(getattr(configs, 'exo_dim', 0))
        self.exo_head = None
        if self.exo_dim > 0:
            self._build_exo_head(self.exo_dim)

    def _build_exo_head(self, E: int):
        self.exo_head = nn.Sequential(
            nn.Linear(E, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.exo_dim = int(E)

    @staticmethod
    def _apply_eol_prior(y: torch.Tensor, future_exo: torch.Tensor, idx: int, strength: float = 0.2) -> torch.Tensor:
        """
        간단한 EOL prior:
          future_exo[:, :, idx]를 표준화한 후 (증가할수록 감소 편향) 가산/감산
        """
        so = future_exo[:, :, idx].float()              # [B,H]
        so_n = (so - so.mean(dim=1, keepdim=True)) / (so.std(dim=1, keepdim=True) + 1e-6)
        return y - strength * so_n

    def forward(self,
                x: torch.Tensor,
                future_exo: torch.Tensor | None = None,
                *,
                part_ids: torch.Tensor | None = None,
                exo_is_normalized: bool | None = None
                ) -> torch.Tensor:
        """
        x: [B,L,C], future_exo: [B,H,E], part_ids: [B]
        return: [B,H]
        """
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        # 1) 정규화
        x = self.revin(x, 'norm')                 # [B,L,C]
        z = self.backbone(x)                      # [B,D]

        # (옵션) part embedding 결합
        if (self.part_emb is not None) and (part_ids is not None):
            pe = self.part_emb(part_ids)          # [B,P]
            z = self.z_fuser(torch.cat([z, pe], dim=1))

        # 2) 확장
        x_bhf = self.expander(z)                  # [B,H,F]
        x_bhf_n = self.pre_ln(x_bhf)              # [B,H,F]

        B, H = z.size(0), self.horizon
        t = torch.linspace(-1, 1, H, device=z.device).unsqueeze(0)

        # 3) base + α
        b = self.base_head_b(z)                   # [B,1]
        m = self.base_head_m(z)                   # [B,1]
        base = b + m * t                          # [B,H]
        alpha = torch.sigmoid(self.base_gate(z)).expand(-1, H)  # [B,H]

        # 4) residual
        resid = self.head(x_bhf_n).squeeze(-1)    # [B,H]
        resid = self.resid_scale * resid
        resid = resid - resid.mean(dim=1, keepdim=True)

        # 5) step gate
        xg = self.gate_ln(x_bhf_n).transpose(1, 2)    # [B,F,H]
        g1 = self.gate_act(self.gate_conv_3(xg))
        g2 = self.gate_act(self.gate_conv_5(xg))
        g3 = self.gate_act(self.gate_conv_d3(xg))
        gcat = torch.cat([g1, g2, g3], dim=1)         # [B,96,H]
        gcat = self.gate_do(gcat)
        g_logit = self.gate_reduce(gcat).transpose(1, 2).squeeze(-1)  # [B,H]

        tau = torch.linspace(-1.0, 1.0, H, device=x_bhf.device).view(1, H).expand(B, H)
        g_logit = (g_logit + self.tau_weight * tau + self.g_bias)
        g_logit = torch.clamp(self.g_gain * (g_logit / self.gate_temp), -self.g_logit_clip, self.g_logit_clip)
        gate = torch.sigmoid(g_logit)  # [B,H]
        gate = gate - gate.mean(dim=1, keepdim=True) + 0.5
        gate = torch.clamp(gate, 0.05, 0.95)

        # 6) 혼합
        y = alpha * base + (1.0 - alpha) * (gate * resid)          # [B,H]

        # 7) exogenous(정규화 공간 가산 or denorm 후 가산)
        ex = None
        if future_exo is not None:
            if (self.exo_head is None) or (future_exo.size(-1) != self.exo_dim):
                new_E = int(future_exo.size(-1))
                if self.training:
                    print(f"[PatchMixer/BaseModel][warn] exo_dim mismatch "
                          f"(model={self.exo_dim}, batch={new_E}). Rebuilding exo_head.")
                self._build_exo_head(new_E)

            ex = apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon=self.horizon,
                out_dtype=y.dtype,
                out_device=y.device
            )
            if exo_is_normalized:
                y = y + ex

        # 8) EOL prior
        if self.use_eol_prior and (future_exo is not None) and (self.eol_feature_index < future_exo.size(-1)):
            y = self._apply_eol_prior(y, future_exo, self.eol_feature_index, strength=0.2)

        # 9) scale/bias + DW 곡률
        y = y * self.out_scale + self.out_bias
        yc = self.dw_head(y.unsqueeze(1)).squeeze(1)
        y  = y + self.dw_gain * yc

        # 10) 역정규화 + (필요 시) 원단위 exogenous 가산
        y = self.revin(y.unsqueeze(-1), 'denorm').squeeze(-1)
        if (ex is not None) and (not exo_is_normalized):
            y = y + ex

        # 11) 추론 시 음수 clamp
        if self.final_nonneg and (not self.training):
            y = torch.clamp_min(y, 0.0)

        return y


# -------------------------
# PatchMixer + Decomposition Quantile Head (Q=3)
# -------------------------
class QuantileModel(nn.Module):
    """
    Multi-Scale PatchMixer Backbone + DecompositionQuantileHead
    output: {'q': (B, 3, H)}
    + (옵션) part embedding, EOL prior, final_nonneg 등
    """
    def __init__(self, configs: PatchMixerConfig):
        super().__init__()
        self.is_quantile = True
        self.model_name = 'PatchMixer QuantileModel'
        self.configs = configs

        self.horizon = configs.horizon
        self.exo_dim = int(getattr(configs, 'exo_dim', 0))
        self.f_out = int(getattr(configs, 'expander_f_out', 128))
        self.n_harmonics = int(getattr(configs, 'expander_n_harmonics', 8))
        self.final_nonneg = bool(getattr(configs, 'final_nonneg', True))
        self.use_eol_prior = bool(getattr(configs, 'use_eol_prior', False))
        self.eol_feature_index = int(getattr(configs, 'eol_feature_index', 0))
        self.exo_is_normalized_default = bool(getattr(configs, 'exo_is_normalized_default', True))

        # 1) Backbone (멀티스케일)
        self.patch_cfgs = getattr(configs, 'patch_cfgs', ())
        self.per_branch_dim = int(getattr(configs, 'per_branch_dim', 64))
        self.fused_dim = int(getattr(configs, 'fused_dim', 128))
        self.fusion = getattr(configs, 'fusion', 'concat')

        self.backbone = MultiScalePatchMixerBackbone(
            base_configs=configs,
            patch_cfgs=self.patch_cfgs,
            per_branch_dim=self.per_branch_dim,
            fused_dim=self.fused_dim,
            fusion=self.fusion,
        )
        d_in = self.backbone.out_dim

        # (옵션) Part embedding
        self.use_part_embedding = bool(getattr(configs, 'use_part_embedding', False))
        self.part_emb = None
        self.z_fuser = None
        if self.use_part_embedding and int(getattr(configs, 'part_vocab_size', 0)) > 0:
            pdim = int(getattr(configs, 'part_embed_dim', 16))
            self.part_emb = nn.Embedding(int(configs.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(d_in + pdim, d_in)

        # 2) Temporal Expander: [B,D] -> [B,H,F]
        self.expander = TemporalExpander(
            d_in=d_in, horizon=self.horizon, f_out=self.f_out, dropout=float(getattr(configs, 'dropout', 0.1)),
            use_sinus=True,
            season_period=int(getattr(configs, "season_period", 52)),
            max_harmonics=int(getattr(configs, "max_harmonics", 16)),
            use_conv=True
        )

        # 3) Quantile Head
        head_hidden = int(getattr(configs, 'head_hidden', 128))
        self.head = DecompositionQuantileHead(
            in_features=self.f_out,
            quantiles=[0.1, 0.5, 0.9],
            hidden=head_hidden,
            dropout=float(getattr(configs, 'head_dropout', 0.0) or 0.0),
            mid=0.5,
            use_trend=True,
            fourier_k=self.n_harmonics,
            agg="mean",
        )

        # 외생
        self.exo_head = None
        if self.exo_dim > 0:
            self._build_exo_head(self.exo_dim)

        self.revin = RevIN(int(getattr(configs, 'enc_in', 1)))

    def _build_exo_head(self, E: int):
        self.exo_head = nn.Sequential(
            nn.Linear(E, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.exo_dim = int(E)

    @staticmethod
    def _ensure_bqh(q: torch.Tensor, horizon: int, qlen: int) -> torch.Tensor:
        # 허용: (B,Q,H) 또는 (B,H,Q)
        if q.dim() != 3:
            raise ValueError(f"pred must be 3D, got {q.shape}")
        B, A, Bdim = q.shape
        if A == qlen and Bdim == horizon:  # (B,Q,H)
            return q
        if A == horizon and Bdim == qlen:  # (B,H,Q)
            return q.permute(0, 2, 1).contiguous()
        raise ValueError(f"pred shape must be (B,{qlen},{horizon}) or (B,{horizon},{qlen}), got {q.shape}")

    @staticmethod
    def _apply_eol_prior(q: torch.Tensor, future_exo: torch.Tensor, idx: int, strength: float = 0.2) -> torch.Tensor:
        # q: [B,Q,H]
        so = future_exo[:, :, idx].float()                     # [B,H]
        so_n = (so - so.mean(dim=1, keepdim=True)) / (so.std(dim=1, keepdim=True) + 1e-6)
        return q - strength * so_n.unsqueeze(1)                # 모든 분위에 동일 적용

    def forward(self,
                x: torch.Tensor,
                future_exo: torch.Tensor | None = None,
                *,
                part_ids: torch.Tensor | None = None,
                exo_is_normalized: bool | None = None,
                **kwargs):
        """
        x: (B, L, N), future_exo: (B, H, E), part_ids: (B,)
        return: {"q": (B, 3, H)}   # RevIN denorm 완료, (추론 시) clamp_nonneg 옵션 적용
        """
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        # 0) 입력 정규화
        x_n = self.revin(x, 'norm')

        # 1) 백본 → [B, D]
        z = self.backbone(x_n)

        # (옵션) part embedding 결합
        if (self.part_emb is not None) and (part_ids is not None):
            pe = self.part_emb(part_ids)          # [B,P]
            z = self.z_fuser(torch.cat([z, pe], dim=1))

        # 2) 시점 확장
        x_bhf = self.expander(z)                  # (B, H, F)

        # 3) 분위수 예측(정규화 공간)
        q = self.head(x_bhf)                      # (B, 3, H)
        q = self._ensure_bqh(q, self.horizon, qlen=3)

        # 4) exogenous shift (정규화 공간/원단위 중 선택)
        ex = None
        if future_exo is not None:
            if (self.exo_head is None) or (future_exo.size(-1) != self.exo_dim):
                new_E = int(future_exo.size(-1))
                if self.training:
                    print(f"[PatchMixer/QuantileModel][warn] exo_dim mismatch "
                          f"(model={self.exo_dim}, batch={new_E}). Rebuilding exo_head.")
                self._build_exo_head(new_E)

            ex = apply_exo_shift_linear(self.exo_head, future_exo,
                                        horizon=self.horizon,
                                        out_dtype=q.dtype, out_device=q.device)
            if exo_is_normalized:
                q = q + ex.unsqueeze(1)

        # 5) (옵션) EOL prior (정규화 공간에서 보정)
        if self.use_eol_prior and (future_exo is not None) and (self.eol_feature_index < future_exo.size(-1)):
            q = self._apply_eol_prior(q, future_exo, self.eol_feature_index, strength=0.2)

        # 6) RevIN 역정규화 (분위수별)
        qs = []
        for i in range(q.size(1)):
            qi = self.revin(q[:, i, :].unsqueeze(-1), 'denorm').squeeze(-1)  # [B,H]
            qs.append(qi.unsqueeze(1))  # [B,1,H]
        q_raw = torch.cat(qs, dim=1)    # [B,3,H]

        # 7) 원단위 exogenous 가산(정규화 공간에서 더하지 않았다면)
        if (ex is not None) and (not exo_is_normalized):
            q_raw = q_raw + ex.unsqueeze(1)

        # 8) 추론 시 음수 clamp
        if self.final_nonneg and (not self.training):
            q_raw = torch.clamp_min(q_raw, 0.0)

        return {"q": q_raw}
