from __future__ import annotations
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "TitanBaseModel",
    "TitanLMMModel",
    "TitanSeq2SeqModel",
]

from modeling_module.models.common_layers.RevIN import RevIN

# --- 패키지/로컬 모두 대응 가능한 유연한 임포트 ---
try:
    from modeling_module.models.Titan.backbone import MemoryEncoder
    from modeling_module.models.Titan.common.decoder import TitanDecoder
except Exception:
    from backbone import MemoryEncoder          # type: ignore
    from decoder import TitanDecoder            # type: ignore

# TitanConfig 는 configs.py 에 정의되어 있다고 가정
try:
    from modeling_module.models.Titan.common.configs import TitanConfig
except Exception:
    try:
        from configs import TitanConfig         # type: ignore
    except Exception:
        TitanConfig = None                      # type: ignore


def _merge_cfg_kwargs(cfg_obj, **kwargs):
    """cfg(dataclass)와 kwargs 병합. kwargs 우선."""
    cfg_dict = asdict(cfg_obj) if cfg_obj is not None else {}
    cfg_dict.update(kwargs)
    return cfg_dict


class _TitanBase(nn.Module):
    """
    공통 베이스: config 보관 + Encoder/Decoder 조립
    - RevIN, past exo 주입, final clamp 등 공통 옵션 관리
    """
    def __init__(self, *, config: Optional["TitanConfig"] = None, **kwargs):
        super().__init__()
        params = _merge_cfg_kwargs(config, **kwargs)

        # Core
        self.input_dim: int = int(params["input_dim"])
        self.lookback: int = int(params["lookback"])
        self.horizon: int = int(params["horizon"])
        self.d_model: int = int(params.get("d_model", 256))
        self.n_layers: int = int(params.get("n_layers", 3))
        self.n_heads: int = int(params.get("n_heads", 4))
        self.d_ff: int = int(params.get("d_ff", 512))
        self.dropout: float = float(params.get("dropout", 0.1))

        # RevIN 옵션 (configs.py에 이미 존재)
        self.use_revin = bool(params.get("use_revin", True))
        self.revin_subtract_last = bool(params.get("revin_subtract_last", False))
        self.revin_affine = bool(params.get("revin_affine", True))
        self.revin_use_std = bool(params.get("revin_use_std", True))

        # Memory/LMM
        self.contextual_mem_size: int = int(params.get("contextual_mem_size", 256))
        self.persistent_mem_size: int = int(params.get("persistent_mem_size", 64))

        # Exogenous (future)
        self.use_exogenous: bool = bool(params.get("use_exogenous", False))
        self.exo_dim: int = int(params.get("exo_dim", 0))
        self.use_calendar_exo: bool = bool(params.get("use_calendar_exo", False))

        # Past exogenous 주입 모드 (PatchMixer와 동일 인터페이스)
        #   - 'z_gate': encoder output memory에서 summary z에 gate로 주입
        #   - 'fuse_input': 입력단에서 concat 후 Linear → [B,L,input_dim]
        #   - 'none': 사용 안 함
        self.past_exo_mode: str = str(params.get("past_exo_mode", "none")).lower()
        self.use_past_exo: bool = self.past_exo_mode != "none"

        # 입력단 & 카테고리 exo 유틸 (PatchMixer 스타일)
        self._in_fuser: Optional[nn.Linear] = None      # [C_total] -> [input_dim]
        self._cat_embs: Optional[nn.ModuleList] = None
        self._cat_table_sizes: Optional[list[int]] = None
        self._cat_embed_dims: Optional[list[int]] = None

        # z-level 주입을 위한 proj/gate
        self._z_exo_proj: Optional[nn.Linear] = None    # [E_sum] -> [D]
        self._z_gate: Optional[nn.Linear] = None        # [D] -> [D]

        # Output constraint
        self.final_clamp_nonneg: bool = bool(params.get("final_clamp_nonneg", True))

        # Seq2Seq decoder 전용 파라미터(필요 시 사용)
        self.dec_layers: int = int(params.get("dec_layers", 2))
        self.dec_heads: int = int(params.get("dec_heads", 4))
        self.dec_d_ff: int = int(params.get("dec_d_ff", 512))
        self.dec_dropout: float = float(params.get("dec_dropout", 0.1))

        # 원본 config 보관(트레이너가 참조 가능)
        self.config = config

        # RevIN: 타깃 채널만 정규화/복원
        self.target_channel = int(params.get("target_channel", 0))
        self.revin = RevIN(
            num_features=1,  # 단일 타깃 채널 기준
            affine=self.revin_affine,
            subtract_last=self.revin_subtract_last,
            use_std=self.revin_use_std,
        ) if self.use_revin else None

        # Encoder
        self.encoder = MemoryEncoder(
            self.input_dim,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.d_ff,
            self.contextual_mem_size,
            self.persistent_mem_size,
            self.dropout,
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: "TitanConfig"):
        return cls(config=config)

    def _clamp(self, y: torch.Tensor) -> torch.Tensor:
        """최종 출력 비음수 제약(옵션)."""
        return y.clamp_min(0) if self.final_clamp_nonneg else y

    # ----------------- RevIN -----------------
    # 입력 x: [B, L, C] -> RevIN(norm) -> [B, L, C]
    def _maybe_revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        if (self.revin is None) or (x.size(-1) == 0):
            return x
        tc = self.target_channel
        x_t = x[:, :, tc:tc+1]
        x_t = self.revin(x_t, mode="norm")   # [B,L,1]
        x = x.clone()
        x[:, :, tc:tc+1] = x_t
        return x

    # 출력 y: [B, H] -> RevIN(denorm) -> [B, H]
    def _maybe_revin_denorm(self, y: torch.Tensor) -> torch.Tensor:
        if (self.revin is None) or (y.dim() != 2):
            return y
        y_in = y.unsqueeze(-1)                 # [B, H, 1]
        y_out = self.revin(y_in, mode="denorm")
        return y_out.squeeze(-1)              # [B, H]

    # ----------------- past exo: 카테고리 임베딩 -----------------
    def _maybe_build_cat_embeds(self, K: int, *, device):
        if self._cat_embs is None:
            # 각 카테고리 feature 마다 16차원 embedding, 초기 테이블 크기 256
            self._cat_embs = nn.ModuleList(
                [nn.Embedding(256, 16) for _ in range(K)]
            ).to(device)
            self._cat_table_sizes = [256] * K
            self._cat_embed_dims = [16] * K

    def _ensure_cat_capacity(self, j: int, max_id: int, device):
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

    # ----------------- past exo: 입력단 결합 (fuse_input 모드) -----------------
    def _fuse_inputs_input_level(
        self,
        x: torch.Tensor,                          # [B,L,C]
        pe_cont: Optional[torch.Tensor],          # [B,L,E_c]
        pe_cat: Optional[torch.Tensor],           # [B,L,E_k] (long)
    ) -> torch.Tensor:
        """
        PatchMixer의 fuse_input 모드와 동일한 형태:
          - x, past_exo_cont, past_exo_cat(임베딩) concat → Linear → [B,L,input_dim]
        """
        B, L, C = x.shape
        feats = [x]

        if pe_cont is not None and pe_cont.numel() > 0:
            feats.append(pe_cont.float())         # [B,L,E_c]

        if pe_cat is not None and pe_cat.numel() > 0:
            E_k = pe_cat.size(-1)
            self._maybe_build_cat_embeds(E_k, device=x.device)
            embs = []
            for j in range(E_k):
                ids = pe_cat[..., j].clamp_min(0).long()
                self._ensure_cat_capacity(j, int(ids.max().item()), device=x.device)
                embs.append(self._cat_embs[j](ids))   # [B,L,d_j]
            feats.append(torch.cat(embs, dim=-1))     # [B,L,sum d_j]

        fused = torch.cat(feats, dim=-1)              # [B,L,C_total]

        # input_dim으로 projection (Titan encoder input)
        if (self._in_fuser is None
                or self._in_fuser.in_features != fused.size(-1)
                or self._in_fuser.out_features != self.input_dim):
            self._in_fuser = nn.Linear(fused.size(-1), self.input_dim, bias=True).to(x.device)

        return self._in_fuser(fused)                  # [B,L,input_dim]

    # ----------------- past exo: z-level 결합 (z_gate 모드) -----------------
    def _inject_exo_to_memory(
        self,
        memory: torch.Tensor,                         # [B,L,D]
        pe_cont: Optional[torch.Tensor],              # [B,L,E_c]
        pe_cat: Optional[torch.Tensor],               # [B,L,E_k]
    ) -> torch.Tensor:
        """
        PatchMixer의 z_gate 모드를 Titan에 맞게 변형:
          - memory를 시계열 축으로 평균 → z: [B,D]
          - past exo를 평균 pool → exo_vec: [B,E_sum]
          - exo_vec -> proj -> [B,D], z -> gate -> [B,D]
          - mem_exo = z + gate * exo_z
          - memory 전체 토큰에 같은 보정 벡터를 더함: memory + mem_exo.unsqueeze(1)
        """
        if ((pe_cont is None) or pe_cont.numel() == 0) and ((pe_cat is None) or pe_cat.numel() == 0):
            return memory

        B, L, D = memory.shape
        feats = []

        if pe_cont is not None and pe_cont.numel() > 0:
            # [B,L,E_c] → mean → [B,E_c]
            feats.append(pe_cont.float().mean(dim=1))

        if pe_cat is not None and pe_cat.numel() > 0:
            E_k = pe_cat.size(-1)
            self._maybe_build_cat_embeds(E_k, device=memory.device)
            embs = []
            for j in range(E_k):
                ids = pe_cat[..., j].clamp_min(0).long()
                self._ensure_cat_capacity(j, int(ids.max().item()), device=memory.device)
                emb_j = self._cat_embs[j](ids)        # [B,L,d]
                embs.append(emb_j.mean(dim=1))        # [B,d]
            feats.append(torch.cat(embs, dim=-1))     # [B,sum d]

        if not feats:
            return memory

        exo_vec = torch.cat(feats, dim=-1)            # [B,E_sum]

        # proj/gate 준비
        if (self._z_exo_proj is None
                or self._z_exo_proj.in_features != exo_vec.size(-1)
                or self._z_exo_proj.out_features != D):
            self._z_exo_proj = nn.Linear(exo_vec.size(-1), D, bias=True).to(memory.device)

        if (self._z_gate is None
                or self._z_gate.in_features != D
                or self._z_gate.out_features != D):
            self._z_gate = nn.Linear(D, D, bias=True).to(memory.device)

        # memory summary z: [B,D]
        z = memory.mean(dim=1)                        # [B,D]
        exo_z = self._z_exo_proj(exo_vec)             # [B,D]
        gate = torch.sigmoid(self._z_gate(z))         # [B,D]

        mem_exo = z + gate * exo_z                    # [B,D]
        memory = memory + mem_exo.unsqueeze(1)        # [B,L,D]
        return memory


class TitanBaseModel(_TitanBase):
    """
    Encoder-only: TitanDecoder를 수평(H) 차원 투영기로 사용하고 Linear로 1채널 예측.
    - future_exo: [B,H,E] → TitanDecoder로 전달
    - past_exo_cont/past_exo_cat: PatchMixer 스타일로 입력단 또는 memory 단에 주입
    """
    def __init__(self, *, config: Optional["TitanConfig"] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.decoder = TitanDecoder(
            d_model=self.d_model,
            n_layers=1,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            horizon=self.horizon,
            exo_dim=(self.exo_dim if self.use_exogenous else 0),
        )
        self.proj = nn.Linear(self.d_model, 1)

        # 디버깅용 옵션 출력
        print(f"[TitanBaseModel] revin_use_std={self.revin_use_std}, "
              f"revin_subtract_last={self.revin_subtract_last}, "
              f"final_clamp_nonneg={self.final_clamp_nonneg}, "
              f"past_exo_mode={self.past_exo_mode}")

    def forward(
        self,
        x: torch.Tensor,                              # [B,L,C]
        *,
        future_exo: Optional[torch.Tensor] = None,    # [B,H,E]
        past_exo_cont: Optional[torch.Tensor] = None, # [B,L,E_c]
        past_exo_cat: Optional[torch.Tensor] = None,  # [B,L,E_k]
        part_ids: Optional[torch.Tensor] = None,      # [B] (현재 버전은 사용 안 함)
        **kwargs,
    ) -> torch.Tensor:
        # 0) past exo fuse_input 모드: 입력단에서 concat 후 linear
        if self.use_past_exo and self.past_exo_mode == "fuse_input" and (
            (past_exo_cont is not None) or (past_exo_cat is not None)
        ):
            x_in = self._fuse_inputs_input_level(x, past_exo_cont, past_exo_cat)
        else:
            x_in = x

        # 1) RevIN norm (입력 전처리)
        x_n = self._maybe_revin_norm(x_in)           # [B,L,C] (C=input_dim)

        # 2) Encoder
        memory = self.encoder(x_n)                   # [B,L,D]

        # 2.5) past exo z_gate 모드: memory에 summary exo 주입
        if self.use_past_exo and self.past_exo_mode == "z_gate" and (
            (past_exo_cont is not None) or (past_exo_cat is not None)
        ):
            memory = self._inject_exo_to_memory(memory, past_exo_cont, past_exo_cat)

        # 3) Decoder
        dec = self.decoder(memory, future_exo)       # [B,H,D]
        y = self.proj(dec).squeeze(-1)               # [B,H]

        # 4) RevIN denorm (출력 복원)
        y = self._maybe_revin_denorm(y)              # [B,H]

        # 5) 최종 제약(비음수 등)
        return self._clamp(y)


class TitanLMMModel(_TitanBase):
    """
    LMM 특화 디코딩이 필요하면 TitanDecoder 내부에서 분기하도록 구성(여기선 공용 Decoder 사용).
    - past/future exo 주입 패턴은 TitanBaseModel과 동일.
    """
    def __init__(self, *, config: Optional["TitanConfig"] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.decoder = TitanDecoder(
            d_model=self.d_model,
            n_layers=1,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            horizon=self.horizon,
            exo_dim=(self.exo_dim if self.use_exogenous else 0),
        )
        self.proj = nn.Linear(self.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,                              # [B,L,C]
        *,
        future_exo: Optional[torch.Tensor] = None,    # [B,H,E]
        past_exo_cont: Optional[torch.Tensor] = None, # [B,L,E_c]
        past_exo_cat: Optional[torch.Tensor] = None,  # [B,L,E_k]
        part_ids: Optional[torch.Tensor] = None,      # [B]
        **kwargs,
    ) -> torch.Tensor:
        if self.use_past_exo and self.past_exo_mode == "fuse_input" and (
            (past_exo_cont is not None) or (past_exo_cat is not None)
        ):
            x_in = self._fuse_inputs_input_level(x, past_exo_cont, past_exo_cat)
        else:
            x_in = x

        x_n = self._maybe_revin_norm(x_in)
        memory = self.encoder(x_n)

        if self.use_past_exo and self.past_exo_mode == "z_gate" and (
            (past_exo_cont is not None) or (past_exo_cat is not None)
        ):
            memory = self._inject_exo_to_memory(memory, past_exo_cont, past_exo_cat)

        dec = self.decoder(memory, future_exo)
        y = self.proj(dec).squeeze(-1)
        y = self._maybe_revin_denorm(y)
        return self._clamp(y)


class TitanSeq2SeqModel(_TitanBase):
    """
    Seq2Seq: 다층 디코더를 이용하여 미래 H 단계의 컨텍스트를 생성 후 1채널 예측.
    - past/future exo 주입 패턴 동일.
    """
    def __init__(self, *, config: Optional["TitanConfig"] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.decoder = TitanDecoder(
            d_model=self.d_model,
            n_layers=self.dec_layers,
            n_heads=self.dec_heads,
            d_ff=self.dec_d_ff,
            dropout=self.dec_dropout,
            horizon=self.horizon,
            exo_dim=(self.exo_dim if self.use_exogenous else 0),
        )
        self.proj = nn.Linear(self.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,                              # [B,L,C]
        *,
        future_exo: Optional[torch.Tensor] = None,    # [B,H,E]
        past_exo_cont: Optional[torch.Tensor] = None, # [B,L,E_c]
        past_exo_cat: Optional[torch.Tensor] = None,  # [B,L,E_k]
        part_ids: Optional[torch.Tensor] = None,      # [B]
        **kwargs,
    ) -> torch.Tensor:
        if self.use_past_exo and self.past_exo_mode == "fuse_input" and (
            (past_exo_cont is not None) or (past_exo_cat is not None)
        ):
            x_in = self._fuse_inputs_input_level(x, past_exo_cont, past_exo_cat)
        else:
            x_in = x

        x_n = self._maybe_revin_norm(x_in)
        memory = self.encoder(x_n)

        if self.use_past_exo and self.past_exo_mode == "z_gate" and (
            (past_exo_cont is not None) or (past_exo_cat is not None)
        ):
            memory = self._inject_exo_to_memory(memory, past_exo_cont, past_exo_cat)

        dec = self.decoder(memory, future_exo)
        y = self.proj(dec).squeeze(-1)
        y = self._maybe_revin_denorm(y)
        return self._clamp(y)
