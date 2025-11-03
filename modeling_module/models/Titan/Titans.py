from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from modeling_module.models.Titan.backbone import MemoryEncoder, PatchMemoryEncoder
from modeling_module.models.Titan.common.configs import TitanConfig
from modeling_module.models.Titan.common.decoder import TitanDecoder
from modeling_module.models.Titan.common.memory import LMM
from modeling_module.models.common_layers.RevIN import RevIN
from modeling_module.models.common_layers.TrendCorrector import TrendCorrector


def _ensure_config_instance(config):
    return config() if isinstance(config, type) else config


# =========================
# Titan Base (point model)
# =========================
class Model(nn.Module):
    """
    Titan BaseModel
      - RevIN: norm-only inside model; NO denorm here (forecaster denorms)
      - Encoder: MemoryEncoder
      - Head: Linear(D->H) (+ Softplus if nonneg_head=True)
      - Optional exogenous term (time-distributed)
    """
    is_quantile: bool = False

    def __init__(self, config: TitanConfig):
        super().__init__()
        cfg = _ensure_config_instance(config)

        self.model_name = "Titan BaseModel"
        self.horizon = cfg.horizon

        # RevIN (norm only)
        self.revin_layer = RevIN(num_features=cfg.input_dim, affine=True, subtract_last=True)

        # Backbone
        self.encoder = MemoryEncoder(
            cfg.input_dim, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff,
            cfg.contextual_mem_size, cfg.persistent_mem_size
        )

        # Head (nonneg option)
        nonneg = getattr(cfg, "nonneg_head", True)
        head = [nn.Linear(cfg.d_model, cfg.horizon)]
        if nonneg:
            head.append(nn.Softplus())
        self.output_proj = nn.Sequential(*head)

        # Optional exogenous term
        self.exo_dim = getattr(cfg, "exo_dim", 0)
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1)  # → [B,H,1] then squeeze
            )
        else:
            self.exo_head = None

        # Optional final clamp (safety net)
        self.final_clamp_nonneg = getattr(cfg, "final_clamp_nonneg", True)

    def forward(
        self,
        x: torch.Tensor,                          # [B,L,C]
        future_exo: Optional[torch.Tensor] = None,  # [B,H,exo_dim] or None
        mode: Optional[str] = None,
    ) -> torch.Tensor:                            # returns [B,H] in NORMALIZED space
        # 1) RevIN normalize
        x_n = self.revin_layer(x, "norm")         # [B,L,C]

        # 2) Encode
        enc = self.encoder(x_n)                   # [B,L,D]
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return [B,L,D], got {tuple(enc.shape)}")

        # 3) Head (last token)
        y_n = self.output_proj(enc[:, -1, :])     # [B,H]

        # 4) Optional exogenous addend
        if (self.exo_head is not None) and (future_exo is not None):
            if future_exo.dim() != 3 or future_exo.size(1) != self.horizon:
                raise ValueError(
                    f"future_exo must be [B,H,exo_dim] with H={self.horizon}, got {tuple(future_exo.shape)}"
                )
            exo_term = self.exo_head(future_exo).squeeze(-1)  # [B,H]
            y_n = y_n + exo_term

        # 5) Optional final clamp in normalized space (keeps ≥0 after Softplus + addends)
        if self.final_clamp_nonneg:
            y_n = torch.clamp_min(y_n, 0.0)

        # NOTE: NO denorm here — forecaster handles it consistently.
        return y_n


# =========================
# Titan + LMM
# =========================
class LMMModel(nn.Module):
    """
    Titan + LMM
      - RevIN: norm-only inside model; NO denorm here
      - Encoder: MemoryEncoder
      - LMM: Local Memory Matching
      - Head: Linear(D->H) (+ Softplus if nonneg_head=True)
      - TrendCorrector
      - Optional exogenous term
    """
    is_quantile: bool = False

    def __init__(self, config: TitanConfig):
        super().__init__()
        cfg = _ensure_config_instance(config)

        self.model_name = "Titan LMMModel"
        self.horizon = cfg.horizon

        self.revin_layer = RevIN(num_features=cfg.input_dim, affine=True, subtract_last=True)

        self.encoder = MemoryEncoder(
            cfg.input_dim, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff,
            cfg.contextual_mem_size, cfg.persistent_mem_size
        )

        self.lmm = LMM(d_model=cfg.d_model, top_k=getattr(cfg, "lmm_top_k", 5))

        nonneg = getattr(cfg, "nonneg_head", True)
        head = [nn.Linear(cfg.d_model, cfg.horizon)]
        if nonneg:
            head.append(nn.Softplus())
        self.output_proj = nn.Sequential(*head)

        self.trend_corrector = TrendCorrector(d_model=cfg.d_model, out_dim=cfg.horizon)

        self.exo_dim = getattr(cfg, "exo_dim", 0)
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1)
            )
        else:
            self.exo_head = None

        self.final_clamp_nonneg = getattr(cfg, "final_clamp_nonneg", True)

    def _collect_memories(self, enc: torch.Tensor) -> torch.Tensor:
        """
        Assemble memory for LMM: contextual + persistent + encoded.
        Returns [B,M,D].
        """
        B, L, D = enc.shape
        mem_chunks = []

        # contextual memory (latest available)
        ctx = None
        for layer in self.encoder.layers:
            m = getattr(layer.attn, "contextual_memory", None)
            if m is not None and m.numel() > 0:
                ctx = m
        if ctx is not None:
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(B, -1, -1)  # [B,M,D]
            elif ctx.dim() == 3 and ctx.size(0) != B:
                ctx = ctx.expand(B, -1, -1)
            mem_chunks.append(ctx)

        # persistent memory (from last layer)
        pm = getattr(self.encoder.layers[-1].attn, "persistent_memory", None)
        if pm is not None and pm.numel() > 0:
            mem_chunks.append(pm.unsqueeze(0).expand(B, -1, -1))  # [B,M,D]

        # encoded tokens
        mem_chunks.append(enc)

        memory = torch.cat(mem_chunks, dim=1)  # [B,M_tot,D]
        return memory

    def forward(
        self,
        x: torch.Tensor,                          # [B,L,C]
        future_exo: Optional[torch.Tensor] = None,  # [B,H,exo_dim] or None
        mode: str = "train",
    ) -> torch.Tensor:                            # returns [B,H] in NORMALIZED space
        # 1) RevIN normalize
        x_n = self.revin_layer(x, "norm")

        # 2) Encode
        enc = self.encoder(x_n)                   # [B,L,D]
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return [B,L,D], got {tuple(enc.shape)}")

        # 3) LMM augmentation
        memory = self._collect_memories(enc)      # [B,M,D]
        enhanced = self.lmm(enc, memory)          # [B,L,D]

        # 4) Head (last token)
        y_core_n = self.output_proj(enhanced[:, -1, :])  # [B,H]

        # 5) Trend correction (vector input)
        y_n = y_core_n + self.trend_corrector(enhanced[:, -1, :])  # [B,H]

        # 6) Optional exogenous term
        if (self.exo_head is not None) and (future_exo is not None):
            if future_exo.dim() != 3 or future_exo.size(1) != self.horizon:
                raise ValueError(
                    f"future_exo must be [B,H,exo_dim] with H={self.horizon}, got {tuple(future_exo.shape)}"
                )
            exo_term = self.exo_head(future_exo).squeeze(-1)  # [B,H]
            y_n = y_n + exo_term

        if self.final_clamp_nonneg:
            y_n = torch.clamp_min(y_n, 0.0)

        return y_n


# =========================
# Titan Patch + LMM
# =========================
class PatchLMMModel(nn.Module):
    """
    Patch-based Titan + LMM
      - RevIN: norm-only inside model; NO denorm here
      - Encoder: PatchMemoryEncoder
      - LMM
      - Head (+ Softplus if nonneg_head=True)
      - TrendCorrector
      - Optional exogenous term
    """
    is_quantile: bool = False

    def __init__(self, config: TitanConfig):
        super().__init__()
        cfg = _ensure_config_instance(config)

        self.model_name = "Titan PatchLMMModel"
        self.horizon = getattr(cfg, "horizon", getattr(cfg, "output_horizon", None))
        assert self.horizon is not None, "config.horizon (or output_horizon) is required"

        self.revin_layer = RevIN(num_features=cfg.input_dim, affine=True, subtract_last=True)

        # Patch-based encoder
        self.encoder = PatchMemoryEncoder(
            input_dim=cfg.input_dim,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            contextual_mem_size=cfg.contextual_mem_size,
            persistent_mem_size=cfg.persistent_mem_size,
            patch_len=getattr(cfg, "patch_len", 12),
            patch_stride=getattr(cfg, "patch_stride", 8),
            n_mixer_blocks=getattr(cfg, "n_mixer_blocks", 2),
            mixer_hidden=getattr(cfg, "mixer_hidden", 2 * cfg.d_model),
            mixer_kernel=getattr(cfg, "mixer_kernel", 7),
            dropout=getattr(cfg, "dropout", 0.1),
        )

        self.lmm = LMM(d_model=cfg.d_model, top_k=getattr(cfg, "lmm_top_k", 5))

        nonneg = getattr(cfg, "nonneg_head", True)
        head = [nn.Linear(cfg.d_model, self.horizon)]
        if nonneg:
            head.append(nn.Softplus())
        self.output_proj = nn.Sequential(*head)

        self.trend_corrector = TrendCorrector(d_model=cfg.d_model, out_dim=self.horizon)

        # Which memory to use for LMM: 'encoded' | 'context'
        self.lmm_memory_source = getattr(cfg, "lmm_memory_source", "encoded")

        self.exo_dim = getattr(cfg, "exo_dim", 0)
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1)
            )
        else:
            self.exo_head = None

        self.final_clamp_nonneg = getattr(cfg, "final_clamp_nonneg", True)

    def _get_lmm_memory(self, enc: torch.Tensor) -> torch.Tensor:
        """Return [B,M,D] memory tensor for LMM."""
        B, Lp, D = enc.shape
        if self.lmm_memory_source == "context":
            ctx = None
            for layer in self.encoder.layers:
                mem = getattr(layer.attn, "contextual_memory", None)
                if mem is not None and mem.numel() > 0:
                    ctx = mem  # [M,D]
            if ctx is not None:
                return ctx.unsqueeze(0).expand(B, -1, -1)  # [B,M,D]
        return enc  # [B,L',D]

    def forward(
        self,
        x: torch.Tensor,                           # [B,L,C]
        future_exo: Optional[torch.Tensor] = None, # [B,H,exo_dim] or None
        mode: str = "train",
    ) -> torch.Tensor:                             # returns [B,H] in NORMALIZED space
        # 1) RevIN normalize
        x_n = self.revin_layer(x, "norm")

        # 2) Encode
        enc = self.encoder(x_n)                    # [B,L',D]
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return [B,L,D], got {tuple(enc.shape)}")

        # 3) LMM
        memory = self._get_lmm_memory(enc)         # [B,M,D]
        if memory.dim() == 2:
            memory = memory.unsqueeze(0).expand(enc.size(0), -1, -1)
        enhanced = self.lmm(enc, memory)           # [B,L',D]

        # 4) Head
        y_core_n = self.output_proj(enhanced[:, -1, :])   # [B,H]

        # 5) Trend (vector input)
        y_n = y_core_n + self.trend_corrector(enhanced[:, -1, :])  # [B,H]

        # 6) Optional exogenous addend
        if (self.exo_head is not None) and (future_exo is not None):
            if future_exo.dim() != 3 or future_exo.size(1) != self.horizon:
                raise ValueError(
                    f"future_exo must be [B,H,exo_dim] with H={self.horizon}, got {tuple(future_exo.shape)}"
                )
            exo_term = self.exo_head(future_exo).squeeze(-1)  # [B,H]
            y_n = y_n + exo_term

        if self.final_clamp_nonneg:
            y_n = torch.clamp_min(y_n, 0.0)

        return y_n


# =========================
# Titan LMM Seq2Seq
# =========================
class LMMSeq2SeqModel(nn.Module):
    """
    Titan LMM Seq2Seq
      - RevIN: norm-only inside model; NO denorm here
      - Encoder: MemoryEncoder
      - Decoder: TitanDecoder (causal)
      - Head: time-distributed Linear→(Softplus optional)→squeeze
      - TrendCorrector
      - Optional exogenous term
    """
    is_quantile: bool = False

    def __init__(self, config: TitanConfig):
        super().__init__()
        cfg = _ensure_config_instance(config)

        self.model_name = "Titan LMMSeq2SeqModel"
        self.horizon = getattr(cfg, "horizon", getattr(cfg, "output_horizon", None))
        assert self.horizon is not None, "config.horizon is required"

        self.revin_layer = RevIN(num_features=cfg.input_dim, affine=True, subtract_last=True)

        self.encoder = MemoryEncoder(
            cfg.input_dim, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff,
            cfg.contextual_mem_size, cfg.persistent_mem_size
        )

        n_dec_layers = getattr(cfg, "n_dec_layers", 1)
        dec_dropout = getattr(cfg, "dec_dropout", 0.1)
        exo_dim = getattr(cfg, "exo_dim", 0)

        self.decoder = TitanDecoder(
            d_model=cfg.d_model,
            n_layers=n_dec_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=dec_dropout,
            horizon=self.horizon,
            exo_dim=exo_dim,
            causal=True
        )

        nonneg = getattr(cfg, "nonneg_head", True)
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.d_model, 1),
            nn.Softplus() if nonneg else nn.Identity()
        )

        self.trend_corrector = TrendCorrector(d_model=cfg.d_model, out_dim=self.horizon)

        self.exo_dim = getattr(cfg, "exo_dim", 0)
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1)
            )
        else:
            self.exo_head = None

        self.final_clamp_nonneg = getattr(cfg, "final_clamp_nonneg", True)

    def forward(
        self,
        x: torch.Tensor,                              # [B,L,C]
        future_exo: Optional[torch.Tensor] = None,    # [B,H,exo_dim] or None
        mode: str = "train",
    ) -> torch.Tensor:                                # returns [B,H] in NORMALIZED space
        # 1) RevIN normalize
        x_n = self.revin_layer(x, "norm")

        # 2) Encode
        enc = self.encoder(x_n)                       # [B,L,D]
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return [B,L,D], got {tuple(enc.shape)}")

        # 3) Decode (causal)
        dec = self.decoder(enc, future_exo)           # [B,H,D]

        # 4) Time-distributed head
        y_core_n = self.output_proj(dec).squeeze(-1)  # [B,H]

        # 5) Trend (vector input)
        y_n = y_core_n + self.trend_corrector(enc[:, -1, :])  # [B,H]

        # 6) Optional exogenous addend
        if (self.exo_head is not None) and (future_exo is not None):
            if future_exo.dim() != 3 or future_exo.size(1) != self.horizon:
                raise ValueError(
                    f"future_exo must be [B,H,exo_dim] with H={self.horizon}, got {tuple(future_exo.shape)}"
                )
            exo_term = self.exo_head(future_exo).squeeze(-1)  # [B,H]
            y_n = y_n + exo_term

        if self.final_clamp_nonneg:
            y_n = torch.clamp_min(y_n, 0.0)

        return y_n


# =========================
# Titan FeatureModel
# =========================
class FeatureModel(nn.Module):
    """
    Titan FeatureModel
      - Encoder: MemoryEncoder
      - Simple feature projection and fusion (add)
      - Head: Linear(D -> H)
      - RevIN: not applied here by design
    """
    is_quantile: bool = False

    def __init__(self, config: TitanConfig, feature_dim: int = 7):
        super().__init__()
        cfg = _ensure_config_instance(config)

        self.model_name = "Titan FeatureModel"
        self.horizon = cfg.horizon

        self.encoder = MemoryEncoder(
            cfg.input_dim, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff,
            cfg.contextual_mem_size, cfg.persistent_mem_size
        )

        self.feature_proj = nn.Linear(feature_dim, cfg.d_model)
        self.output_proj = nn.Linear(cfg.d_model, cfg.horizon)

    def forward(self, x: torch.Tensor, feature_x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)                        # [B,L,D]
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return [B,L,D], got {tuple(enc.shape)}")

        feat = self.feature_proj(feature_x).unsqueeze(1)  # [B,1,D]
        combined = enc[:, -1, :] + feat.squeeze(1)       # [B,D]
        return self.output_proj(combined)                # [B,H]


# =========================
# Test-Time Memory Manager
# =========================
class TestTimeMemoryManager:
    """
    Lightweight TTA (Test-Time Adaptation) helper for Titan-family models.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    @torch.no_grad()
    def add_context(self, new_context: torch.Tensor) -> None:
        """
        Push contextual memory to every encoder layer (MAC).
        """
        device = next(self.model.parameters()).device
        new_context = new_context.to(device).detach()
        if not hasattr(self.model, "encoder") or not hasattr(self.model.encoder, "layers"):
            return
        for block in self.model.encoder.layers:
            if hasattr(block, "attn") and hasattr(block.attn, "update_contextual_memory"):
                block.attn.update_contextual_memory(new_context)

    def adapt(self, x_new: torch.Tensor, y_new: torch.Tensor, steps: int = 1) -> float:
        """
        Simple supervised TTA loop (use sparingly).
        """
        device = next(self.model.parameters()).device
        x_new = x_new.to(device).float()
        y_new = y_new.to(device).float()

        self.model.train()
        last_loss = 0.0
        for _ in range(steps):
            pred = self.model(x_new)
            loss = self.loss_fn(pred, y_new)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.item())
        self.model.eval()
        return last_loss
