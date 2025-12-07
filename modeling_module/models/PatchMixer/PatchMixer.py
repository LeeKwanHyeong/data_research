from typing import List, Tuple, Optional
import torch
import torch.nn as nn

from modeling_module.models.PatchMixer.backbone import PatchMixerBackbone, MultiScalePatchMixerBackbone
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.common_layers.RevIN import RevIN
from modeling_module.models.common_layers.heads.quantile_heads.decomposition_quantile_head import DecompositionQuantileHead
from modeling_module.utils.exogenous_utils import apply_exo_shift_linear
from modeling_module.utils.temporal_expander import TemporalExpander


# -------------------------
# small helpers
# -------------------------
def _nearest_odd(k: int) -> int:
    return k if k % 2 == 1 else (k + 1)

def make_patch_cfgs(lookback: int, n_branches: int = 3) -> List[Tuple[int, int, int]]:
    assert lookback >= 8, f"lookback={lookback} is too small (>=8)."
    fracs = [1/4, 1/2, 3/4][:n_branches]
    raw = [max(4, min(lookback, int(round(lookback * f)))) for f in fracs]
    P = sorted(list(dict.fromkeys(raw)))
    cfgs = []
    for i, p in enumerate(P):
        s = max(1, p // 2)
        k = [3, 5, 7][min(i, 2)]
        k = _nearest_odd(k)
        cfgs.append((p, s, k))
    return cfgs


# =====================================================================
# PatchMixer → Horizon regression (Point)   [BaseModel]
# =====================================================================
class BaseModel(nn.Module):
    """
    PatchMixer Backbone → TemporalExpander → per-step head
    + base(절편+기울기, α-게이트) + step-gate(Conv1d+τ) + DW residual
    + (옵션) part embedding, EOL prior, final_nonneg 등
    + (신규) 과거 외생 주입 모드:
        - past_exo_mode="z_gate"(기본): RevIN(x) 후, z 단계에서 과거 외생을 게이팅 결합
        - past_exo_mode="fuse_input": 입력단 concat→Linear→enc_in → RevIN
    """
    def __init__(self, configs: PatchMixerConfig):
        super().__init__()
        self.model_name = 'PatchMixer BaseModel'
        self.configs = configs

        self.horizon = int(configs.horizon)
        self.f_out = int(getattr(configs, 'f_out', 128))

        # ----- past exo 주입 구성 -----
        self.past_exo_mode = str(getattr(configs, 'past_exo_mode', 'z_gate'))  # 'z_gate' | 'fuse_input' | 'none'
        self.use_past_exo  = self.past_exo_mode.lower() != 'none'
        self._in_fuser: Optional[nn.Linear] = None       # (입력단 concat 모드용) [C_total] -> [enc_in]
        self._cat_embs: Optional[nn.ModuleList] = None   # 카테고리별 임베딩
        self._cat_table_sizes: Optional[list[int]] = None
        self._cat_embed_dims: Optional[list[int]] = None

        # z단 결합용(권장 모드): 과거 exo를 요약해 z 차원과 결합
        self._z_exo_proj: Optional[nn.Linear] = None     # [E_sum] -> [z_dim]
        self._z_gate: Optional[nn.Linear] = None         # z 게이트

        # ----- backbone / common -----
        self.backbone = PatchMixerBackbone(configs=configs)
        self.exo_is_normalized_default = bool(getattr(configs, 'exo_is_normalized_default', True))
        self.final_nonneg = bool(getattr(configs, 'final_nonneg', True))
        self.use_eol_prior = bool(getattr(configs, 'use_eol_prior', False))
        self.eol_feature_index = int(getattr(configs, 'eol_feature_index', 0))

        # 백본 out 차원
        self.in_dim = getattr(self.backbone, 'out_dim', getattr(self.backbone, 'patch_repr_dim', None))
        assert self.in_dim is not None, "Backbone must expose out_dim or patch_repr_dim."

        # z 정합 및 expander
        self.z_align: Optional[nn.Linear] = None
        self.z_proj: nn.Module = nn.Identity()
        self.expander: Optional[TemporalExpander] = None

        # (옵션) Part Embedding
        self.use_part_embedding = bool(getattr(configs, 'use_part_embedding', False))
        self.part_emb = None
        self.z_fuser = None
        if self.use_part_embedding and int(getattr(configs, 'part_vocab_size', 0)) > 0:
            pdim = int(getattr(configs, 'part_embed_dim', 16))
            self.part_emb = nn.Embedding(int(configs.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(self.in_dim + pdim, self.in_dim)

        # Temporal Expander: [B,D] -> [B,H,F]
        self.expander = TemporalExpander(
            d_in=self.in_dim,
            horizon=self.horizon,
            f_out=self.f_out,
            dropout=float(getattr(configs, 'dropout', 0.1)),
            use_sinus=True,
            season_period=int(getattr(configs, 'expander_season_period', 52)),
            max_harmonics=int(getattr(configs, 'expander_max_harmonics', 16)),
            use_conv=True
        )

        # RevIN (norm-only)
        self.revin = RevIN(int(getattr(configs, 'enc_in', 1)))

        # base(절편 + 기울기) + base gate α
        self.base_head_b = nn.Linear(self.in_dim, 1)
        self.base_head_m = nn.Linear(self.in_dim, 1)
        self.base_gate   = nn.Linear(self.in_dim, 1)
        nn.init.constant_(self.base_gate.bias, -2.5)

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

        # 미래 외생(head)
        self.exo_dim = int(getattr(configs, 'exo_dim', 0))
        self.exo_head = None
        if self.exo_dim > 0:
            self._build_exo_head(self.exo_dim)

    # ----- 카테고리 임베딩 유틸 -----
    def _maybe_build_cat_embeds(self, K: int, *, device):
        if self._cat_embs is None:
            self._cat_embs = nn.ModuleList([nn.Embedding(256, 16) for _ in range(K)]).to(device)
            self._cat_table_sizes = [256]*K
            self._cat_embed_dims  = [16]*K

    def _ensure_cat_capacity(self, j: int, max_id: int, device):
        assert self._cat_embs is not None and self._cat_table_sizes is not None
        if max_id < self._cat_table_sizes[j]:
            return
        old = self._cat_embs[j]; old_num, dim = old.num_embeddings, old.embedding_dim
        new_num = max(max_id + 1, old_num * 2)
        new = nn.Embedding(new_num, dim).to(device)
        with torch.no_grad():
            new.weight[:old_num].copy_(old.weight)
        self._cat_embs[j] = new
        self._cat_table_sizes[j] = new_num

    # ----- 입력단 결합(fuse_input 모드) -----
    def _fuse_inputs_input_level(self, x, pe_cont, pe_cat):
        """
        x: [B,L,C], pe_cont: [B,L,E_c] or None, pe_cat: [B,L,E_k] (long) or None
        return: [B,L,enc_in]
        """
        B, L, C = x.shape
        feats = [x]
        if pe_cont is not None and pe_cont.numel() > 0:
            feats.append(pe_cont.float())
        if pe_cat is not None and pe_cat.numel() > 0:
            E_k = pe_cat.size(-1)
            self._maybe_build_cat_embeds(E_k, device=x.device)
            embs = []
            for j in range(E_k):
                ids = pe_cat[..., j].clamp_min(0).long()
                self._ensure_cat_capacity(j, int(ids.max().item()), device=x.device)
                embs.append(self._cat_embs[j](ids))  # [B,L,d_j]
            feats.append(torch.cat(embs, dim=-1))    # [B,L,sum d_j]
        fused = torch.cat(feats, dim=-1)

        enc_in = int(getattr(self.configs, 'enc_in', 1))
        if (self._in_fuser is None) or (self._in_fuser.in_features != fused.size(-1)) or (self._in_fuser.out_features != enc_in):
            self._in_fuser = nn.Linear(fused.size(-1), enc_in, bias=True).to(x.device)
        return self._in_fuser(fused)

    # ----- z단 결합(z_gate 모드) -----
    def _fuse_inputs_z_level(self, z, pe_cont, pe_cat):
        """
        pe_cont: [B,L,E_c], pe_cat: [B,L,E_k]
        - 시계열 축 평균(pool) 후 concat → [B, E_sum] → proj to [B, z_dim]
        - z_gate로 게이트(sigmoid) → z + gate * exo_proj
        """
        if (pe_cont is None or pe_cont.numel() == 0) and (pe_cat is None or pe_cat.numel() == 0):
            return z

        B = z.size(0)
        feats = []
        if (pe_cont is not None) and pe_cont.numel() > 0:
            # [B,L,E_c] → 평균풀링 → [B,E_c]
            feats.append(pe_cont.float().mean(dim=1))

        if (pe_cat is not None) and pe_cat.numel() > 0:
            E_k = pe_cat.size(-1)
            self._maybe_build_cat_embeds(E_k, device=z.device)
            embs = []
            # 각 카테고리 feature별로 [B,L]→임베딩[B,L,d]→평균[B,d]
            for j in range(E_k):
                ids = pe_cat[..., j].clamp_min(0).long()
                self._ensure_cat_capacity(j, int(ids.max().item()), device=z.device)
                emb_j = self._cat_embs[j](ids)      # [B,L,d]
                embs.append(emb_j.mean(dim=1))      # [B,d]
            feats.append(torch.cat(embs, dim=-1))    # [B,sum d]

        exo_vec = torch.cat(feats, dim=-1) if len(feats) > 0 else None
        if exo_vec is None:
            return z

        z_dim = z.size(-1)
        if (self._z_exo_proj is None) or (self._z_exo_proj.in_features != exo_vec.size(-1)) or (self._z_exo_proj.out_features != z_dim):
            self._z_exo_proj = nn.Linear(exo_vec.size(-1), z_dim, bias=True).to(z.device)

        if (self._z_gate is None) or (self._z_gate.in_features != z_dim) or (self._z_gate.out_features != z_dim):
            self._z_gate = nn.Linear(z_dim, z_dim, bias=True).to(z.device)

        exo_z = self._z_exo_proj(exo_vec)           # [B, z_dim]
        gate  = torch.sigmoid(self._z_gate(z))      # [B, z_dim]
        return z + gate * exo_z

    # ----- 미래 exo head -----
    def _build_exo_head(self, E: int, device: Optional[torch.device] = None):
        dev = device if device is not None else (next(self.parameters()).device if any(True for _ in self.parameters()) else 'cpu')
        self.exo_head = nn.Sequential(
            nn.Linear(E, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        ).to(dev)
        self.exo_dim = int(E)

    @staticmethod
    def _apply_eol_prior(y: torch.Tensor, future_exo: torch.Tensor, idx: int, strength: float = 0.2) -> torch.Tensor:
        so = future_exo[:, :, idx].float()              # [B,H]
        so_n = (so - so.mean(dim=1, keepdim=True)) / (so.std(dim=1, keepdim=True) + 1e-6)
        return y - strength * so_n

    def forward(
        self,
        x: torch.Tensor,                                 # [B,L,C]
        future_exo: Optional[torch.Tensor] = None,       # [B,H,E]
        *,
        past_exo_cont: Optional[torch.Tensor] = None,    # [B,L,E_c]
        past_exo_cat: Optional[torch.Tensor] = None,     # [B,L,E_k] (long)
        part_ids: Optional[torch.Tensor] = None,         # [B]
        exo_is_normalized: Optional[bool] = None,
        **kwargs
    ) -> torch.Tensor:
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        # 0) 과거 exo 주입 분기
        if self.use_past_exo and self.past_exo_mode.lower() == 'fuse_input' and (past_exo_cont is not None or past_exo_cat is not None):
            x_in = self._fuse_inputs_input_level(x, past_exo_cont, past_exo_cat)
        else:
            x_in = x

        # 1) RevIN + Backbone
        if getattr(self.configs, 'use_revin', True):  # Config 확인
            x_n = self.revin(x_in, 'norm')
        else:
            x_n = x_in
        z = self.backbone(x_n)  # [B, D_eff]

        # z 정합(동적 변화 대비)
        if (self.z_align is None) or (z.size(-1) != self.in_dim):
            self.z_align = nn.Linear(z.size(-1), self.in_dim, bias=False).to(z.device)
        z = self.z_align(z)  # [B, in_dim]

        # 1.5) (권장) z단에서 과거 exo 결합
        if self.use_past_exo and self.past_exo_mode.lower() == 'z_gate' and (past_exo_cont is not None or past_exo_cat is not None):
            z = self._fuse_inputs_z_level(z, past_exo_cont, past_exo_cat)

        # (옵션) part embedding 결합
        if (self.part_emb is not None) and (part_ids is not None) and torch.is_tensor(part_ids):
            pe = self.part_emb(part_ids)  # [B, pdim]
            if (self.z_fuser is None) or (self.z_fuser.in_features != (z.size(-1) + pe.size(-1))):
                self.z_fuser = nn.Linear(z.size(-1) + pe.size(-1), self.in_dim, bias=True).to(z.device)
            z = self.z_fuser(torch.cat([z, pe], dim=1))

        # 2) Expander
        if self.expander is None or getattr(self.expander, "d_in", None) != z.size(-1):
            self.expander = TemporalExpander(
                d_in=z.size(-1), horizon=self.horizon, f_out=self.f_out,
                dropout=float(getattr(self.configs, 'dropout', 0.1)),
                use_sinus=True,
                season_period=int(getattr(self.configs, 'expander_season_period', 52)),
                max_harmonics=int(getattr(self.configs, 'expander_max_harmonics', 16)),
                use_conv=True
            ).to(z.device)

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
                    print(f"[PatchMixer/BaseModel][warn] exo_dim mismatch (model={self.exo_dim}, batch={new_E}). Rebuilding exo_head.")
                self._build_exo_head(new_E)

            ex = apply_exo_shift_linear(
                self.exo_head, future_exo,
                horizon=self.horizon, out_dtype=y.dtype, out_device=y.device
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
        if getattr(self.configs, 'use_revin', True):  # Config 확인
            y = self.revin(y.unsqueeze(-1), 'denorm').squeeze(-1)
        if (ex is not None) and (not exo_is_normalized):
            y = y + ex

        # 11) 추론 시 음수 clamp
        if self.final_nonneg and (not self.training):
            y = torch.clamp_min(y, 0.0)

        return y


# =====================================================================
# PatchMixer + Decomposition Quantile Head (Q=3)   [QuantileModel]
# =====================================================================
class QuantileModel(nn.Module):
    """
    Multi-Scale PatchMixer Backbone + TemporalExpander + DecompositionQuantileHead
    출력: {"q": (B, 3, H)}  # RevIN denorm 후, 추론 시 음수 clamp(옵션) 적용
    (신규) 과거 외생 주입 모드 동일 지원: past_exo_mode in {'z_gate','fuse_input','none'}
    """
    def __init__(self, configs: PatchMixerConfig):
        super().__init__()
        self.is_quantile = True
        self.model_name = "PatchMixer QuantileModel"
        self.configs = configs

        # ----- 기본 하이퍼 -----
        self.horizon = int(configs.horizon)
        self.f_out = int(getattr(configs, "f_out", 128))
        self.n_harmonics = int(getattr(configs, "expander_n_harmonics", 8))

        # 플래그
        self.final_nonneg = bool(getattr(configs, "final_nonneg", True))
        self.use_eol_prior = bool(getattr(configs, "use_eol_prior", False))
        self.eol_feature_index = int(getattr(configs, "eol_feature_index", 0))
        self.exo_is_normalized_default = bool(getattr(configs, "exo_is_normalized_default", True))

        # ----- past exo 주입 구성 -----
        self.past_exo_mode = str(getattr(configs, 'past_exo_mode', 'z_gate'))
        self.use_past_exo  = self.past_exo_mode.lower() != 'none'
        self._in_fuser: Optional[nn.Linear] = None
        self._cat_embs: Optional[nn.ModuleList] = None
        self._cat_table_sizes: Optional[list[int]] = None
        self._cat_embed_dims: Optional[list[int]] = None
        self._z_exo_proj: Optional[nn.Linear] = None
        self._z_gate: Optional[nn.Linear] = None

        # ----- 멀티스케일 백본 -----
        self.patch_cfgs = tuple(getattr(configs, "patch_cfgs", ())) or tuple(make_patch_cfgs(configs.lookback, n_branches=3))
        self.per_branch_dim = int(getattr(configs, "per_branch_dim", 64))
        self.fused_dim = int(getattr(configs, "fused_dim", 128))
        self.fusion = getattr(configs, "fusion", "concat")

        self.backbone = MultiScalePatchMixerBackbone(
            base_configs=configs,
            patch_cfgs=self.patch_cfgs,
            per_branch_dim=self.per_branch_dim,
            fused_dim=self.fused_dim,
            fusion=self.fusion,
        )
        self.in_dim = self.backbone.out_dim
        self.z_align: Optional[nn.Linear] = None
        self.expander: Optional[TemporalExpander] = None
        self.z_proj: nn.Module = nn.Identity()

        # (옵션) part embedding
        self.use_part_embedding = bool(getattr(configs, 'use_part_embedding', False))
        self.part_emb = None
        self.z_fuser = None
        if self.use_part_embedding and int(getattr(configs, 'part_vocab_size', 0)) > 0:
            pdim = int(getattr(configs, 'part_embed_dim', 16))
            self.part_emb = nn.Embedding(int(configs.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(self.in_dim + pdim, self.in_dim)

        # expander
        self.expander = TemporalExpander(
            d_in=self.in_dim,
            horizon=self.horizon,
            f_out=self.f_out,
            dropout=float(getattr(configs, 'dropout', 0.1)),
            use_sinus=True,
            season_period=int(getattr(configs, 'expander_season_period', 52)),
            max_harmonics=int(getattr(configs, 'expander_max_harmonics', 16)),
            use_conv=True
        )

        # Quantile Head
        head_hidden = int(getattr(configs, 'head_hidden', 128))
        self.head = DecompositionQuantileHead(
            in_features=int(getattr(configs, 'f_out', 128)),
            quantiles=[0.1, 0.5, 0.9],
            hidden=head_hidden,
            dropout=float(getattr(configs, 'head_dropout', 0.0) or 0.0),
            mid=0.5,
            use_trend=True,
            fourier_k=int(getattr(configs, 'expander_n_harmonics', 8)),
            agg="mean",
        )

        # ----- Exogenous (future) -----
        self.exo_dim = int(getattr(configs, "exo_dim", 0))
        self.exo_head = None
        if self.exo_dim > 0:
            self._build_exo_head(self.exo_dim)

        # ----- RevIN -----
        self.revin = RevIN(int(getattr(configs, "enc_in", 1)))

    # ----- 카테고리 임베딩 유틸 -----
    def _maybe_build_cat_embeds(self, K: int, *, device):
        if self._cat_embs is None:
            self._cat_embs = nn.ModuleList([nn.Embedding(256, 16) for _ in range(K)]).to(device)
            self._cat_table_sizes = [256]*K
            self._cat_embed_dims  = [16]*K

    def _ensure_cat_capacity(self, j: int, max_id: int, device):
        if max_id < self._cat_table_sizes[j]:
            return
        old = self._cat_embs[j]; old_num, dim = old.num_embeddings, old.embedding_dim
        new_num = max(max_id + 1, old_num * 2)
        new = nn.Embedding(new_num, dim).to(device)
        with torch.no_grad():
            new.weight[:old_num].copy_(old.weight)
        self._cat_embs[j] = new
        self._cat_table_sizes[j] = new_num

    def _fuse_inputs_input_level(self, x, pe_cont, pe_cat):
        B, L, C = x.shape
        feats = [x]
        if pe_cont is not None and pe_cont.numel() > 0:
            feats.append(pe_cont.float())
        if pe_cat is not None and pe_cat.numel() > 0:
            E_k = pe_cat.size(-1)
            self._maybe_build_cat_embeds(E_k, device=x.device)
            embs = []
            for j in range(E_k):
                ids = pe_cat[..., j].clamp_min(0).long()
                self._ensure_cat_capacity(j, int(ids.max().item()), device=x.device)
                embs.append(self._cat_embs[j](ids))
            feats.append(torch.cat(embs, dim=-1))
        fused = torch.cat(feats, dim=-1)

        enc_in = int(getattr(self.configs, 'enc_in', 1))
        if (self._in_fuser is None) or (self._in_fuser.in_features != fused.size(-1)) or (self._in_fuser.out_features != enc_in):
            self._in_fuser = nn.Linear(fused.size(-1), enc_in, bias=True).to(x.device)
        return self._in_fuser(fused)

    def _fuse_inputs_z_level(self, z, pe_cont, pe_cat):
        if (pe_cont is None or pe_cont.numel() == 0) and (pe_cat is None or pe_cat.numel() == 0):
            return z

        feats = []
        if (pe_cont is not None) and pe_cont.numel() > 0:
            feats.append(pe_cont.float().mean(dim=1))  # [B,E_c]

        if (pe_cat is not None) and pe_cat.numel() > 0:
            E_k = pe_cat.size(-1)
            self._maybe_build_cat_embeds(E_k, device=z.device)
            embs = []
            for j in range(E_k):
                ids = pe_cat[..., j].clamp_min(0).long()
                self._ensure_cat_capacity(j, int(ids.max().item()), device=z.device)
                emb_j = self._cat_embs[j](ids)     # [B,L,d]
                embs.append(emb_j.mean(dim=1))     # [B,d]
            feats.append(torch.cat(embs, dim=-1))

        exo_vec = torch.cat(feats, dim=-1) if len(feats) > 0 else None
        if exo_vec is None:
            return z

        z_dim = z.size(-1)
        if (self._z_exo_proj is None) or (self._z_exo_proj.in_features != exo_vec.size(-1)) or (self._z_exo_proj.out_features != z_dim):
            self._z_exo_proj = nn.Linear(exo_vec.size(-1), z_dim, bias=True).to(z.device)

        if (self._z_gate is None) or (self._z_gate.in_features != z_dim) or (self._z_gate.out_features != z_dim):
            self._z_gate = nn.Linear(z_dim, z_dim, bias=True).to(z.device)

        exo_z = self._z_exo_proj(exo_vec)           # [B, z_dim]
        gate  = torch.sigmoid(self._z_gate(z))      # [B, z_dim]
        return z + gate * exo_z

    def _build_exo_head(self, E: int):
        self.exo_head = nn.Sequential(
            nn.Linear(E, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.exo_dim = int(E)

    @staticmethod
    def _apply_eol_prior(q: torch.Tensor, future_exo: torch.Tensor, idx: int, strength: float = 0.2) -> torch.Tensor:
        so = future_exo[:, :, idx].float()  # [B,H]
        so_n = (so - so.mean(dim=1, keepdim=True)) / (so.std(dim=1, keepdim=True) + 1e-6)
        return q - strength * so_n.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,                         # (B,L,C)
        future_exo: Optional[torch.Tensor] = None,  # (B,H,E)
        *,
        past_exo_cont: Optional[torch.Tensor] = None,   # (B,L,E_c)
        past_exo_cat: Optional[torch.Tensor] = None,    # (B,L,E_k)
        part_ids: Optional[torch.Tensor] = None,        # (B,)
        exo_is_normalized: Optional[bool] = None,
        **kwargs,
    ):
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        # 0) 과거 exo 입력단/혹은 z단 결합 선택
        if self.use_past_exo and self.past_exo_mode.lower() == 'fuse_input' and (past_exo_cont is not None or past_exo_cat is not None):
            x_in = self._fuse_inputs_input_level(x, past_exo_cont, past_exo_cat)
        else:
            x_in = x

        # 1) RevIN → Backbone
        x_n = self.revin(x_in, 'norm')
        z = self.backbone(x_n)  # [B, D_eff]

        if (self.z_align is None) or (z.size(-1) != self.in_dim):
            self.z_align = nn.Linear(z.size(-1), self.in_dim, bias=False).to(z.device)
        z = self.z_align(z)

        # 1.5) (권장) z단에서 과거 exo 결합
        if self.use_past_exo and self.past_exo_mode.lower() == 'z_gate' and (past_exo_cont is not None or past_exo_cat is not None):
            z = self._fuse_inputs_z_level(z, past_exo_cont, past_exo_cat)

        # (옵션) part embedding 결합
        if (self.part_emb is not None) and (part_ids is not None) and torch.is_tensor(part_ids):
            pe = self.part_emb(part_ids)
            if (self.z_fuser is None) or (self.z_fuser.in_features != (z.size(-1) + pe.size(-1))):
                self.z_fuser = nn.Linear(z.size(-1) + pe.size(-1), self.in_dim, bias=True).to(z.device)
            z = self.z_fuser(torch.cat([z, pe], dim=1))

        # 2) Expander
        if self.expander is None or getattr(self.expander, "d_in", None) != z.size(-1):
            self.expander = TemporalExpander(
                d_in=z.size(-1), horizon=self.horizon, f_out=self.f_out,
                dropout=float(getattr(self.configs, 'dropout', 0.1)),
                use_sinus=True,
                season_period=int(getattr(self.configs, "expander_season_period", 52)),
                max_harmonics=int(getattr(self.configs, "expander_max_harmonics", 16)),
                use_conv=True
            ).to(z.device)

        x_bhf = self.expander(z)  # (B, H, F)
        q = self.head(x_bhf)      # (B, 3, H) or (B, H, 3)

        # 3) exogenous shift (정규화 공간)
        ex = None
        if future_exo is not None:
            if (self.exo_head is None) or (future_exo.size(-1) != self.exo_dim):
                new_E = int(future_exo.size(-1))
                if self.training:
                    print(f"[PatchMixer/QuantileModel][warn] exo_dim mismatch (model={self.exo_dim}, batch={new_E}). Rebuilding exo_head.")
                self._build_exo_head(new_E)

            ex = apply_exo_shift_linear(
                self.exo_head,
                future_exo,
                horizon=self.horizon,
                out_dtype=q.dtype,
                out_device=q.device
            )
            if exo_is_normalized:
                # q: (B,Q,H) or (B,H,Q) → 정규화 공간에서 가산 시 축에 주의
                if q.dim() == 3 and q.shape[1] in (3, 5, 9):        # (B,Q,H)
                    q = q + ex.unsqueeze(1)                         # (B,1,H)
                elif q.dim() == 3 and q.shape[2] in (3, 5, 9):      # (B,H,Q)
                    q = q + ex.unsqueeze(-1)                        # (B,H,1)

        # 4) (옵션) EOL prior
        if self.use_eol_prior and (future_exo is not None) and (self.eol_feature_index < future_exo.size(-1)):
            # q를 (B,Q,H) 정렬하여 prior 적용
            if q.shape[1] in (3, 5, 9):  # (B,Q,H)
                q = self._apply_eol_prior(q, future_exo, self.eol_feature_index, strength=0.2)
            else:                         # (B,H,Q)
                q_bqh = q.transpose(1, 2)  # (B,Q,H)
                q_bqh = self._apply_eol_prior(q_bqh, future_exo, self.eol_feature_index, strength=0.2)
                q = q_bqh.transpose(1, 2)

        # 5) RevIN 역정규화(분위별)
        # q를 (B,Q,H) 정렬
        if q.dim() == 3 and q.shape[2] == self.horizon:  # (B,Q,H)
            qs = []
            for i in range(q.size(1)):
                qi = self.revin(q[:, i, :].unsqueeze(-1), "denorm").squeeze(-1)  # [B,H]
                qs.append(qi.unsqueeze(1))
            q_raw = torch.cat(qs, dim=1)  # [B,Q,H]
        elif q.dim() == 3 and q.shape[1] == self.horizon:  # (B,H,Q)
            q = q.transpose(1, 2)  # (B,Q,H)
            qs = []
            for i in range(q.size(1)):
                qi = self.revin(q[:, i, :].unsqueeze(-1), "denorm").squeeze(-1)
                qs.append(qi.unsqueeze(1))
            q_raw = torch.cat(qs, dim=1)  # (B,Q,H)
        else:
            raise RuntimeError(f"Unexpected quantile shape: {tuple(q.shape)}")

        # 6) 원단위 exogenous 가산(정규화 공간에서 더하지 않았다면)
        if (ex is not None) and (not exo_is_normalized):
            q_raw = q_raw + ex.unsqueeze(1)  # (B,1,H)

        # 7) 추론 시 음수 clamp
        if self.final_nonneg and (not self.training):
            q_raw = torch.clamp_min(q_raw, 0.0)

        return {"q": q_raw}