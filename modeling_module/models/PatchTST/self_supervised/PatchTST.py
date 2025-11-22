from typing import Optional, Sequence, Tuple, List

import torch
from torch import nn

from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.PatchTST.supervised.backbone import SupervisedBackbone
from modeling_module.models.common_layers.RevIN import RevIN
from modeling_module.utils.exogenous_utils import apply_exo_shift_linear


# -----------------------------------------
# 1) Point / Quantile Heads
# -----------------------------------------
class PointHead(nn.Module):
    """
    단일 포인트 예측 헤드.
    Backbone에서 나온 feature sequence (B, L_tok, d_model)을 받아서
    평균 혹은 마지막 토큰을 이용해 (B, horizon) 출력으로 변환.
    """
    def __init__(self, d_model: int, horizon: int, agg: str = "mean"):
        super().__init__()
        self.agg = agg
        self.proj = nn.Linear(d_model, horizon)

    def forward(self, z_bld: torch.Tensor) -> torch.Tensor:
        # z_bld: [B, L_tok, d_model]
        if self.agg == "mean":
            feat = z_bld.mean(dim=1)           # [B, d_model]
        else:
            feat = z_bld[:, -1, :]             # [B, d_model]
        return self.proj(feat)                 # [B, H]


class QuantileHead(nn.Module):
    """
    다중 분위수 예측 헤드.
    (B, L_tok, d_model) -> (B, Q, horizon)
    """
    def __init__(
        self,
        d_model: int,
        horizon: int,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        hidden: int = 128,
        monotonic: bool = True,
    ):
        super().__init__()
        self.Q = len(quantiles)
        self.horizon = horizon
        self.monotonic = monotonic

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * self.Q),
        )

    def forward(self, z_bld: torch.Tensor) -> torch.Tensor:
        """
        z_bld: [B, L_tok, d_model]
        return: q: [B, Q, H]
        """
        B = z_bld.size(0)
        feat = z_bld.mean(dim=1)               # [B, d_model]
        q = self.net(feat).view(B, self.horizon, self.Q)  # [B, H, Q]
        q = q.permute(0, 2, 1).contiguous()               # [B, Q, H]
        if self.monotonic:
            # 분위수 축(Q) 방향으로 단조 증가 정렬
            q, _ = torch.sort(q, dim=1)
        return q


# -----------------------------------------
# 2) 공통 Mixin: exo / cat / part embedding 유틸
# -----------------------------------------
class _ExoCatMixin:
    """
    PatchTST Base/Quantile 공통으로 사용하는:
      - past_exo_mode: {'none','fuse_input','z_gate'}
      - categorical 임베딩
      - part embedding
      - future_exo head
      - EOL prior
    """

    # --- 카테고리 임베딩 유틸 ---
    def _maybe_build_cat_embeds(self, K: int, *, device):
        """
        K: 카테고리 feature 개수 (E_k)
        """
        if getattr(self, "_cat_embs", None) is None:
            self._cat_embs = nn.ModuleList([nn.Embedding(256, 16) for _ in range(K)]).to(device)
            self._cat_table_sizes = [256] * K
            self._cat_embed_dims = [16] * K

    def _ensure_cat_capacity(self, j: int, max_id: int, device):
        """
        j번째 categorical feature에 대해 id 범위가 커지면 embedding 테이블 늘림.
        """
        assert self._cat_embs is not None and self._cat_table_sizes is not None
        if max_id < self._cat_table_sizes[j]:
            return
        old = self._cat_embs[j]
        old_num, dim = old.num_embeddings, old.embedding_dim
        new_num = max(max_id + 1, old_num * 2)
        new = nn.Embedding(new_num, dim).to(device)
        with torch.no_grad():
            new.weight[:old_num].copy_(old.weight)
        self._cat_embs[j] = new
        self._cat_table_sizes[j] = new_num

    # --- 입력단 concat 모드: x, past_exo_cont, past_exo_cat 모두 붙여서 enc_in으로 투영 ---
    def _fuse_inputs_input_level(self, x, pe_cont, pe_cat):
        """
        x:       [B, L, C]
        pe_cont: [B, L, E_c] or None
        pe_cat:  [B, L, E_k] or None (long)
        return:  [B, L, enc_in]
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
            feats.append(torch.cat(embs, dim=-1))     # [B,L,sum d_j]
        fused = torch.cat(feats, dim=-1)              # [B,L,C+...]

        enc_in = int(getattr(self.cfg, "c_in", 1))
        if (self._in_fuser is None) or \
           (self._in_fuser.in_features != fused.size(-1)) or \
           (self._in_fuser.out_features != enc_in):
            self._in_fuser = nn.Linear(fused.size(-1), enc_in, bias=True).to(x.device)
        return self._in_fuser(fused)

    # --- z단 결합 모드: backbone 출력 z_bld에 exo 게이팅 ---
    def _fuse_inputs_z_level(self, z_bld, pe_cont, pe_cat):
        """
        z_bld:   [B, L_tok, D]
        pe_cont: [B, L, E_c]
        pe_cat:  [B, L, E_k]
        - pe_*는 시간축(L)에 대해 평균풀링 후 concat → [B, E_sum]
        - Linear로 D 차원으로 proj 후, gate로 결합
        """
        if (pe_cont is None or pe_cont.numel() == 0) and (pe_cat is None or pe_cat.numel() == 0):
            return z_bld

        B, L_tok, D = z_bld.shape
        feats = []
        if (pe_cont is not None) and pe_cont.numel() > 0:
            feats.append(pe_cont.float().mean(dim=1))  # [B,E_c]

        if (pe_cat is not None) and pe_cat.numel() > 0:
            E_k = pe_cat.size(-1)
            self._maybe_build_cat_embeds(E_k, device=z_bld.device)
            embs = []
            for j in range(E_k):
                ids = pe_cat[..., j].clamp_min(0).long()
                self._ensure_cat_capacity(j, int(ids.max().item()), device=z_bld.device)
                emb_j = self._cat_embs[j](ids)         # [B,L,d_j]
                embs.append(emb_j.mean(dim=1))         # [B,d_j]
            feats.append(torch.cat(embs, dim=-1))      # [B,sum d_j]

        exo_vec = torch.cat(feats, dim=-1) if len(feats) > 0 else None
        if exo_vec is None:
            return z_bld

        if (self._z_exo_proj is None) or \
           (self._z_exo_proj.in_features != exo_vec.size(-1)) or \
           (self._z_exo_proj.out_features != D):
            self._z_exo_proj = nn.Linear(exo_vec.size(-1), D, bias=True).to(z_bld.device)

        # gate는 토큰 평균 z_pool에 기반
        z_pool = z_bld.mean(dim=1)  # [B,D]
        if (self._z_gate is None) or \
           (self._z_gate.in_features != D) or \
           (self._z_gate.out_features != D):
            self._z_gate = nn.Linear(D, D, bias=True).to(z_bld.device)

        exo_z = self._z_exo_proj(exo_vec)            # [B,D]
        gate = torch.sigmoid(self._z_gate(z_pool))   # [B,D]
        # 토큰 차원으로 브로드캐스트
        exo_z_tok = exo_z.unsqueeze(1).expand(-1, L_tok, -1)
        gate_tok = gate.unsqueeze(1).expand(-1, L_tok, -1)

        return z_bld + gate_tok * exo_z_tok          # [B,L_tok,D]

    # --- future exo head ---
    def _build_exo_head(self, E: int, device: Optional[torch.device] = None):
        dev = device if device is not None else next(self.parameters()).device
        self.exo_head = nn.Sequential(
            nn.Linear(E, 64),
            nn.GELU(),
            nn.Linear(64, 1),    # step-wise shift
        ).to(dev)
        self.exo_dim = int(E)

    # --- EOL prior ---
    @staticmethod
    def _apply_eol_prior_point(y: torch.Tensor, future_exo: torch.Tensor, idx: int, strength: float = 0.2) -> torch.Tensor:
        """
        point 예측용: y: [B,H], future_exo: [B,H,E]
        """
        so = future_exo[:, :, idx].float()          # [B,H]
        so_n = (so - so.mean(dim=1, keepdim=True)) / (so.std(dim=1, keepdim=True) + 1e-6)
        return y - strength * so_n

    @staticmethod
    def _apply_eol_prior_quantile(q: torch.Tensor, future_exo: torch.Tensor, idx: int, strength: float = 0.2) -> torch.Tensor:
        """
        quantile 예측용: q: [B,Q,H], future_exo: [B,H,E]
        """
        so = future_exo[:, :, idx].float()          # [B,H]
        so_n = (so - so.mean(dim=1, keepdim=True)) / (so.std(dim=1, keepdim=True) + 1e-6)
        return q - strength * so_n.unsqueeze(1)     # [B,Q,H]


# -----------------------------------------
# 3) PatchTST Point Model
# -----------------------------------------
class PatchTSTPointModel(nn.Module, _ExoCatMixin):
    """
    PatchTST Backbone + Point Head
    - PatchMixer/Titan과 동일한 인터페이스:
      forward(x, future_exo=None, *, past_exo_cont=None, past_exo_cat=None,
              part_ids=None, exo_is_normalized=None, **kwargs)
    - 출력은 정규화 공간의 y_n: [B, H]
      (denorm은 Forecaster에서 model.revin_layer(y, 'denorm')으로 일원화)
    """
    def __init__(self, cfg: PatchTSTConfig, attn_core=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = SupervisedBackbone(cfg, attn_core)
        self.head = PointHead(cfg.d_model, cfg.horizon, agg=getattr(cfg, "head_agg", "mean"))

        self.is_quantile = False
        self.horizon = int(cfg.horizon)
        self.model_name = "PatchTST BaseModel"

        # RevIN: Forecaster 호환용 이름
        self.revin_layer = RevIN(num_features=cfg.c_in)

        # ---- exogenous / prior / flags ----
        self.exo_is_normalized_default = bool(getattr(cfg, "exo_is_normalized_default", True))
        self.final_nonneg = bool(getattr(cfg, "final_nonneg", True))
        self.use_eol_prior = bool(getattr(cfg, "use_eol_prior", False))
        self.eol_feature_index = int(getattr(cfg, "eol_feature_index", 0))

        # future exogenous head
        self.exo_dim = int(getattr(cfg, "exo_dim", 0))
        self.exo_head: Optional[nn.Module] = None
        if self.exo_dim > 0:
            self._build_exo_head(self.exo_dim)

        # past exo 모드
        self.past_exo_mode = str(getattr(cfg, "past_exo_mode", "none")).lower()
        self.use_past_exo = self.past_exo_mode != "none"
        self._in_fuser: Optional[nn.Linear] = None
        self._cat_embs: Optional[nn.ModuleList] = None
        self._cat_table_sizes: Optional[list[int]] = None
        self._cat_embed_dims: Optional[list[int]] = None
        self._z_exo_proj: Optional[nn.Linear] = None
        self._z_gate: Optional[nn.Linear] = None

        # part embedding
        self.use_part_embedding = bool(getattr(cfg, "use_part_embedding", False))
        self.part_emb: Optional[nn.Embedding] = None
        self.z_fuser: Optional[nn.Linear] = None
        if self.use_part_embedding and int(getattr(cfg, "part_vocab_size", 0)) > 0:
            pdim = int(getattr(cfg, "part_embed_dim", 16))
            self.part_emb = nn.Embedding(int(cfg.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(cfg.d_model + pdim, cfg.d_model)

    @classmethod
    def from_config(cls, config: PatchTSTConfig):
        return cls(cfg=config)

    def forward(
        self,
        x_b_l_c: torch.Tensor,                         # [B,L,C]
        future_exo: Optional[torch.Tensor] = None,     # [B,Hm,E]
        *,
        past_exo_cont: Optional[torch.Tensor] = None,  # [B,L,E_c]
        past_exo_cat: Optional[torch.Tensor] = None,   # [B,L,E_k]
        part_ids: Optional[torch.Tensor] = None,       # [B]
        exo_is_normalized: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        H = self.horizon

        # --- future_exo horizon 정렬 (self.horizon 기준으로 맞춤) ---
        if future_exo is not None and future_exo.size(1) != H:
            Hm = future_exo.size(1)
            if Hm > H:
                future_exo = future_exo[:, :H, :]
            else:
                pad = H - Hm
                last = future_exo[:, -1:, :].expand(-1, pad, -1)
                future_exo = torch.cat([future_exo, last], dim=1)

        # --- past exo 입력단 결합 ---
        if self.use_past_exo and self.past_exo_mode == "fuse_input" and \
           (past_exo_cont is not None or past_exo_cat is not None):
            x_in = self._fuse_inputs_input_level(x_b_l_c, past_exo_cont, past_exo_cat)
        else:
            x_in = x_b_l_c

        # 1) RevIN 정규화
        x_n = self.revin_layer(x_in, "norm")            # [B,L,C]
        x_n_b_c_l = x_n.permute(0, 2, 1)                # [B,C,L]

        # 2) Backbone
        z = self.backbone(x_n_b_c_l)                    # [B,L_tok,D]

        # 2.5) z단 past exo 결합
        if self.use_past_exo and self.past_exo_mode == "z_gate" and \
           (past_exo_cont is not None or past_exo_cat is not None):
            z = self._fuse_inputs_z_level(z, past_exo_cont, past_exo_cat)

        # 2.6) part embedding 결합
        if (self.part_emb is not None) and (self.z_fuser is not None) and \
           (part_ids is not None) and torch.is_tensor(part_ids):
            pe = self.part_emb(part_ids)            # [B,pdim]
            B, L_tok, D = z.shape
            pe_exp = pe.unsqueeze(1).expand(-1, L_tok, -1)  # [B,L_tok,pdim]
            fused = torch.cat([z, pe_exp], dim=-1)          # [B,L_tok,D+pdim]
            z = self.z_fuser(fused)                         # [B,L_tok,D]

        # 3) Head: 정규화 공간 예측
        y_n = self.head(z)                                # [B,H]

        # 4) future exo shift (정규화 공간)
        ex = None
        if future_exo is not None:
            if (self.exo_head is None) or (future_exo.size(-1) != self.exo_dim):
                new_E = int(future_exo.size(-1))
                if self.training:
                    print(f"[PatchTST/Base][warn] exo_dim mismatch (model={self.exo_dim}, batch={new_E}). Rebuilding exo_head.")
                self._build_exo_head(new_E, device=y_n.device)

            ex = apply_exo_shift_linear(
                self.exo_head,
                future_exo,
                horizon=H,
                out_dtype=y_n.dtype,
                out_device=y_n.device,
            )
            if exo_is_normalized:
                y_n = y_n + ex

        # 5) EOL prior (정규화 공간)
        if self.use_eol_prior and (future_exo is not None) and \
           (self.eol_feature_index < future_exo.size(-1)):
            y_n = self._apply_eol_prior_point(y_n, future_exo, self.eol_feature_index, strength=0.2)

        # 6) 원단위에서 exogenous 가산이 필요한 경우 (exo_is_normalized=False)
        if (ex is not None) and (not exo_is_normalized):
            y_n = y_n + ex

        # 7) 추론 시 음수 clamp (정규화 공간 기준이나, 보통 RevIN scale이 크지 않으면 대략 보호 효과)
        if self.final_nonneg and (not self.training):
            y_n = torch.clamp_min(y_n, 0.0)

        return y_n


# -----------------------------------------
# 4) PatchTST Quantile Model
# -----------------------------------------
class PatchTSTQuantileModel(nn.Module, _ExoCatMixin):
    """
    PatchTST Backbone + Quantile Head
    - PatchMixer/Titan과 동일한 인터페이스
    - 출력은 정규화 공간의 {"q": q_n} with q_n: [B,Q,H]
      (denorm은 Forecaster에서 model.revin_layer를 사용해 수행)
    """
    def __init__(self, cfg: PatchTSTConfig, attn_core=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = SupervisedBackbone(cfg, attn_core)
        self.head = QuantileHead(
            cfg.d_model,
            cfg.horizon,
            quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)),
            hidden=int(getattr(cfg, "head_hidden", 128)),
            monotonic=bool(getattr(cfg, "quantile_monotonic", True)),
        )

        self.is_quantile = True
        self.horizon = int(cfg.horizon)
        self.model_name = "PatchTST QuantileModel"

        # RevIN
        self.revin_layer = RevIN(num_features=cfg.c_in)

        # exogenous / prior
        self.exo_is_normalized_default = bool(getattr(cfg, "exo_is_normalized_default", True))
        self.final_nonneg = bool(getattr(cfg, "final_nonneg", True))
        self.use_eol_prior = bool(getattr(cfg, "use_eol_prior", False))
        self.eol_feature_index = int(getattr(cfg, "eol_feature_index", 0))

        self.exo_dim = int(getattr(cfg, "exo_dim", 0))
        self.exo_head: Optional[nn.Module] = None
        if self.exo_dim > 0:
            self._build_exo_head(self.exo_dim)

        # past exo
        self.past_exo_mode = str(getattr(cfg, "past_exo_mode", "none")).lower()
        self.use_past_exo = self.past_exo_mode != "none"
        self._in_fuser: Optional[nn.Linear] = None
        self._cat_embs: Optional[nn.ModuleList] = None
        self._cat_table_sizes: Optional[list[int]] = None
        self._cat_embed_dims: Optional[list[int]] = None
        self._z_exo_proj: Optional[nn.Linear] = None
        self._z_gate: Optional[nn.Linear] = None

        # part embedding
        self.use_part_embedding = bool(getattr(cfg, "use_part_embedding", False))
        self.part_emb: Optional[nn.Embedding] = None
        self.z_fuser: Optional[nn.Linear] = None
        if self.use_part_embedding and int(getattr(cfg, "part_vocab_size", 0)) > 0:
            pdim = int(getattr(cfg, "part_embed_dim", 16))
            self.part_emb = nn.Embedding(int(cfg.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(cfg.d_model + pdim, cfg.d_model)

    @classmethod
    def from_config(cls, config: PatchTSTConfig):
        return cls(cfg=config)

    def forward(
        self,
        x_b_l_c: torch.Tensor,                         # [B,L,C]
        future_exo: Optional[torch.Tensor] = None,     # [B,Hm,E]
        *,
        past_exo_cont: Optional[torch.Tensor] = None,  # [B,L,E_c]
        past_exo_cat: Optional[torch.Tensor] = None,   # [B,L,E_k]
        part_ids: Optional[torch.Tensor] = None,       # [B]
        exo_is_normalized: Optional[bool] = None,
        **kwargs,
    ):
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        H = self.horizon

        # --- future_exo horizon 정렬 ---
        if future_exo is not None and future_exo.size(1) != H:
            Hm = future_exo.size(1)
            if Hm > H:
                future_exo = future_exo[:, :H, :]
            else:
                pad = H - Hm
                last = future_exo[:, -1:, :].expand(-1, pad, -1)
                future_exo = torch.cat([future_exo, last], dim=1)

        # --- past exo 입력단 결합 ---
        if self.use_past_exo and self.past_exo_mode == "fuse_input" and \
           (past_exo_cont is not None or past_exo_cat is not None):
            x_in = self._fuse_inputs_input_level(x_b_l_c, past_exo_cont, past_exo_cat)
        else:
            x_in = x_b_l_c

        # 1) RevIN 정규화
        x_n = self.revin_layer(x_in, "norm")            # [B,L,C]
        x_n_b_c_l = x_n.permute(0, 2, 1)                # [B,C,L]

        # 2) Backbone
        z = self.backbone(x_n_b_c_l)                    # [B,L_tok,D]

        # 2.5) z단 past exo 결합
        if self.use_past_exo and self.past_exo_mode == "z_gate" and \
           (past_exo_cont is not None or past_exo_cat is not None):
            z = self._fuse_inputs_z_level(z, past_exo_cont, past_exo_cat)

        # 2.6) part embedding
        if (self.part_emb is not None) and (self.z_fuser is not None) and \
           (part_ids is not None) and torch.is_tensor(part_ids):
            pe = self.part_emb(part_ids)                # [B,pdim]
            B, L_tok, D = z.shape
            pe_exp = pe.unsqueeze(1).expand(-1, L_tok, -1)
            fused = torch.cat([z, pe_exp], dim=-1)
            z = self.z_fuser(fused)                     # [B,L_tok,D]

        # 3) Head → 정규화 공간 quantile
        q_n = self.head(z)                              # [B,Q,H]

        # 4) future exo shift (정규화 공간)
        ex = None
        if future_exo is not None:
            if (self.exo_head is None) or (future_exo.size(-1) != self.exo_dim):
                new_E = int(future_exo.size(-1))
                if self.training:
                    print(f"[PatchTST/Quantile][warn] exo_dim mismatch (model={self.exo_dim}, batch={new_E}). Rebuilding exo_head.")
                self._build_exo_head(new_E, device=q_n.device)

            ex = apply_exo_shift_linear(
                self.exo_head,
                future_exo,
                horizon=H,
                out_dtype=q_n.dtype,
                out_device=q_n.device,
            )
            if exo_is_normalized:
                q_n = q_n + ex.unsqueeze(1)             # [B,1,H]

        # 5) EOL prior
        if self.use_eol_prior and (future_exo is not None) and \
           (self.eol_feature_index < future_exo.size(-1)):
            q_n = self._apply_eol_prior_quantile(q_n, future_exo, self.eol_feature_index, strength=0.2)

        # 6) 원단위에서 exogenous 가산
        if (ex is not None) and (not exo_is_normalized):
            q_n = q_n + ex.unsqueeze(1)                 # [B,1,H]

        # 7) 추론 시 음수 clamp
        if self.final_nonneg and (not self.training):
            q_n = torch.clamp_min(q_n, 0.0)

        return {"q": q_n}