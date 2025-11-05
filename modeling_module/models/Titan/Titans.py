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
from modeling_module.models.common_layers.heads.expander_head import ExpanderHead


def _ensure_config_instance(config):
    return config() if isinstance(config, type) else config


def _denorm_forecast_from_revin(
    revin: RevIN,
    y_h: torch.Tensor,                  # [B,H] (RevIN 공간)
    *,
    hist_x: Optional[torch.Tensor] = None,
    ch: int = 0,
    base_mix: float = 0.5,              # ← mean~last 혼합 가중(0~1)
    floor_ratio: float = 0.0,
    floor_min: float = 0.0
) -> torch.Tensor:
    mean  = getattr(revin, "_cached_mean", None)
    std   = getattr(revin, "_cached_std",  None)
    last  = getattr(revin, "_cached_last", None)

    if (revin.subtract_last and last is None) or ((not revin.subtract_last) and (mean is None)):
        return y_h

    device, dtype = y_h.device, y_h.dtype
    if revin.subtract_last:
        base = last[..., ch].squeeze(1).to(device=device, dtype=dtype)
    else:
        m = mean[..., ch].squeeze(1).to(device=device, dtype=dtype)
        if last is not None:
            l = last[..., ch].squeeze(1).to(device=device, dtype=dtype)
            w = float(base_mix)
            base = w * l + (1.0 - w) * m
        else:
            base = m

    if revin.use_std:
        st = std[..., ch].squeeze(1).to(device=device, dtype=dtype)
        if (hist_x is not None) and (floor_ratio > 0.0 or floor_min > 0.0):
            raw_std = hist_x[..., ch].std(dim=1).to(dtype=dtype)
            floor = torch.clamp(floor_ratio * raw_std, min=floor_min)
            st = torch.maximum(st, floor)
        return y_h * st.unsqueeze(1) + base.unsqueeze(1)
    else:
        return y_h + base.unsqueeze(1)

# =========================
# Titan Base (point model)
# =========================
class Model(nn.Module):
    """
    Titan BaseModel
      - RevIN: norm-only inside model; NO direct self.revin(...,'denorm') on [B,H]
      - Encoder: MemoryEncoder
      - Head: ExpanderHead (기본) or Linear(+Softplus)
      - Optional exogenous term (time-distributed)
    """
    is_quantile: bool = False

    def __init__(self, config: TitanConfig):
        super().__init__()
        cfg = _ensure_config_instance(config)

        self.model_name = "Titan BaseModel"
        self.horizon = cfg.horizon

        # 간헐수요 안전: 센터링 전용(표준편차 사용 X)
        self.revin = RevIN(
            num_features=cfg.input_dim,
            affine=True,
            subtract_last=False,
            use_std=False,                 # ← 핵심: Titan만 센터링 전용
        )

        # Backbone
        self.encoder = MemoryEncoder(
            cfg.input_dim, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff,
            cfg.contextual_mem_size, cfg.persistent_mem_size
        )

        self.use_temporal_expander = getattr(cfg, "use_temporal_expander", True)
        if self.use_temporal_expander:
            self.output_head = ExpanderHead(
                d_model=cfg.d_model,
                horizon=cfg.horizon,
                f_out=getattr(cfg, 'expander_f_out', 128),
                nonneg=getattr(cfg, 'nonneg_head', True),
                use_sinus=getattr(cfg, 'expander_use_sinus', True),
                season_period=getattr(cfg, 'expander_season_period', 52),
                max_harmonics=getattr(cfg, 'expander_max_harmonics', 16),
                use_conv=getattr(cfg, 'expander_use_conv', True),
                dropout=getattr(cfg, 'expander_dropout', 0.1)
            )
        else:
            nonneg = getattr(cfg, 'nonneg_head', True)
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

        self.final_clamp_nonneg = getattr(cfg, "final_clamp_nonneg", True)

    def forward(
        self,
        x: torch.Tensor,                             # [B,L,C]
        future_exo: Optional[torch.Tensor] = None,   # [B,H,exo_dim] or None
        mode: Optional[str] = None,
    ) -> torch.Tensor:                               # returns [B,H] (raw scale)
        # 1) RevIN normalize
        x_n = self.revin(x, "norm")                  # [B,L,C]

        # 2) Encode
        enc = self.encoder(x_n)                      # [B,L,D]
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return [B,L,D], got {tuple(enc.shape)}")

        # 3) Head (last token)
        z_last = enc[:, -1, :]
        if self.use_temporal_expander:
            y_n = self.output_head(z_last)           # [B, H] (RevIN 공간)
        else:
            y_n = self.output_proj(z_last)           # [B, H]

        # 4) 안전 denorm ([B,H] 전용)
        y = _denorm_forecast_from_revin(
            self.revin, y_n, hist_x=x, ch=0, floor_ratio=0.0, floor_min=0.0
        )

        # 5) Optional exogenous addend (raw space에서 합산)
        if (self.exo_head is not None) and (future_exo is not None):
            if future_exo.dim() != 3 or future_exo.size(1) != self.horizon:
                raise ValueError(
                    f"future_exo must be [B,H,exo_dim] with H={self.horizon}, got {tuple(future_exo.shape)}"
                )
            exo_term = self.exo_head(future_exo).squeeze(-1)  # [B,H]
            y = y + exo_term

        if self.final_clamp_nonneg:
            y = torch.clamp_min(y, 0.0)

        return y


# =========================
# Titan + LMM
# =========================
class LMMModel(nn.Module):
    is_quantile: bool = False

    def __init__(self, config: TitanConfig):
        super().__init__()
        cfg = _ensure_config_instance(config)

        self.model_name = "Titan LMMModel"
        self.horizon = cfg.horizon

        self.revin = RevIN(
            num_features=cfg.input_dim,
            affine=True,
            subtract_last=False,
            use_std=False,                 # 센터링 전용
        )

        self.encoder = MemoryEncoder(
            cfg.input_dim, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff,
            cfg.contextual_mem_size, cfg.persistent_mem_size
        )

        self.lmm = LMM(d_model=cfg.d_model, top_k=getattr(cfg, "lmm_top_k", 5))

        self.use_temporal_expander = getattr(cfg, "use_temporal_expander", True)
        if self.use_temporal_expander:
            self.output_head = ExpanderHead(
                d_model=cfg.d_model,
                horizon=cfg.horizon,
                f_out=getattr(cfg, 'expander_f_out', 128),
                nonneg=getattr(cfg, 'nonneg_head', True),
                use_sinus=getattr(cfg, 'expander_use_sinus', True),
                season_period=getattr(cfg, 'expander_season_period', 52),
                max_harmonics=getattr(cfg, 'expander_max_harmonics', 16),
                use_conv=getattr(cfg, 'expander_use_conv', True),
                dropout=getattr(cfg, 'expander_dropout', 0.1)
            )
        else:
            nonneg = getattr(cfg, 'nonneg_head', True)
            head = [nn.Linear(cfg.d_model, cfg.horizon)]
            if nonneg:
                head.append(nn.Softplus())
            self.output_proj = nn.Sequential(*head)

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
        x: torch.Tensor,                             # [B,L,C]
        future_exo: Optional[torch.Tensor] = None,   # [B,H,exo_dim] or None
        mode: str = "train",
    ) -> torch.Tensor:                               # [B,H] (raw scale)
        x_n = self.revin(x, "norm")
        enc = self.encoder(x_n)
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return [B,L,D], got {tuple(enc.shape)}")

        memory = self._collect_memories(enc)
        enhanced = self.lmm(enc, memory)
        enhanced = enc + enhanced  # ← residual (추가)
        z_last = enhanced[:, -1, :]
        if self.use_temporal_expander:
            y_core_n = self.output_head(z_last)      # [B,H]
        else:
            y_core_n = self.output_proj(z_last)      # [B,H]

        y = _denorm_forecast_from_revin(self.revin, y_core_n, hist_x=x, ch=0)

        if (self.exo_head is not None) and (future_exo is not None):
            if future_exo.dim() != 3 or future_exo.size(1) != self.horizon:
                raise ValueError(
                    f"future_exo must be [B,H,exo_dim] with H={self.horizon}, got {tuple(future_exo.shape)}"
                )
            exo_term = self.exo_head(future_exo).squeeze(-1)
            y = y + exo_term

        if self.final_clamp_nonneg:
            y = torch.clamp_min(y, 0.0)

        return y


# =========================
# Titan Patch + LMM
# =========================
class PatchLMMModel(nn.Module):
    is_quantile: bool = False

    def __init__(self, config: TitanConfig):
        super().__init__()
        cfg = _ensure_config_instance(config)

        self.model_name = "Titan PatchLMMModel"
        self.horizon = getattr(cfg, "horizon", getattr(cfg, "output_horizon", None))
        assert self.horizon is not None, "config.horizon (or output_horizon) is required"

        self.revin = RevIN(
            num_features=cfg.input_dim,
            affine=True,
            subtract_last=False,
            use_std=False,                 # 센터링 전용
        )

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

        self.use_temporal_expander = getattr(cfg, "use_temporal_expander", True)
        if self.use_temporal_expander:
            self.output_head = ExpanderHead(
                d_model=cfg.d_model,
                horizon=cfg.horizon,
                f_out=getattr(cfg, 'expander_f_out', 128),
                nonneg=getattr(cfg, 'nonneg_head', True),
                use_sinus=getattr(cfg, 'expander_use_sinus', True),
                season_period=getattr(cfg, 'expander_season_period', 52),
                max_harmonics=getattr(cfg, 'expander_max_harmonics', 16),
                use_conv=getattr(cfg, 'expander_use_conv', True),
                dropout=getattr(cfg, 'expander_dropout', 0.1)
            )
        else:
            nonneg = getattr(cfg, 'nonneg_head', True)
            head = [nn.Linear(cfg.d_model, cfg.horizon)]
            if nonneg:
                head.append(nn.Softplus())
            self.output_proj = nn.Sequential(*head)

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
        x: torch.Tensor,                             # [B,L,C]
        future_exo: Optional[torch.Tensor] = None,   # [B,H,exo_dim] or None
        mode: str = "train",
    ) -> torch.Tensor:                               # [B,H] (raw scale)
        x_n = self.revin(x, "norm")
        enc = self.encoder(x_n)
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return [B,L,D], got {tuple(enc.shape)}")

        memory = self._get_lmm_memory(enc)
        if memory.dim() == 2:
            memory = memory.unsqueeze(0).expand(enc.size(0), -1, -1)
        enhanced = self.lmm(enc, memory)

        z_last = enhanced[:, -1, :]
        if self.use_temporal_expander:
            y_core_n = self.output_head(z_last)      # [B,H]
        else:
            y_core_n = self.output_proj(z_last)      # [B,H]

        y = _denorm_forecast_from_revin(self.revin, y_core_n, hist_x=x, ch=0)

        if (self.exo_head is not None) and (future_exo is not None):
            if future_exo.dim() != 3 or future_exo.size(1) != self.horizon:
                raise ValueError(
                    f"future_exo must be [B,H,exo_dim] with H={self.horizon}, got {tuple(future_exo.shape)}"
                )
            exo_term = self.exo_head(future_exo).squeeze(-1)
            y = y + exo_term

        if self.final_clamp_nonneg:
            y = torch.clamp_min(y, 0.0)

        return y


# =========================
# Titan LMM Seq2Seq
# =========================
class LMMSeq2SeqModel(nn.Module):
    is_quantile: bool = False

    def __init__(self, config: TitanConfig):
        super().__init__()
        cfg = _ensure_config_instance(config)

        self.model_name = "Titan LMMSeq2SeqModel"
        self.horizon = getattr(cfg, "horizon", getattr(cfg, "output_horizon", None))
        assert self.horizon is not None, "config.horizon is required"

        self.revin = RevIN(
            num_features=cfg.input_dim,
            affine=True,
            subtract_last=False,
            use_std=False,                 # 센터링 전용
        )

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
    ) -> torch.Tensor:                                # [B,H] (raw scale)
        x_n = self.revin(x, "norm")
        enc = self.encoder(x_n)
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return [B,L,D], got {tuple(enc.shape)}")

        dec = self.decoder(enc, future_exo)           # [B,H,D]

        z_last = enc[:, -1, :]  # [B,D]
        dec = dec + z_last.unsqueeze(1).expand(-1, self.horizon, -1)  # ← residual


        y_core_n = self.output_proj(dec).squeeze(-1)  # [B,H]
        y_core_n = y_core_n + self.trend_corrector(enc[:, -1, :])  # [B,H]

        y = _denorm_forecast_from_revin(self.revin, y_core_n, hist_x=x, ch=0)

        if (self.exo_head is not None) and (future_exo is not None):
            if future_exo.dim() != 3 or future_exo.size(1) != self.horizon:
                raise ValueError(
                    f"future_exo must be [B,H,exo_dim] with H={self.horizon}, got {tuple(future_exo.shape)}"
                )
            exo_term = self.exo_head(future_exo).squeeze(-1)
            y = y + exo_term

        if self.final_clamp_nonneg:
            y = torch.clamp_min(y, 0.0)

        return y


# =========================
# Test-Time Memory Manager
# =========================
class TestTimeMemoryManager:
    """Lightweight TTA helper for Titan-family models."""
    def __init__(self, model: nn.Module, lr: float = 1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    @torch.no_grad()
    def add_context(self, new_context: torch.Tensor) -> None:
        device = next(self.model.parameters()).device
        new_context = new_context.to(device).detach()
        if not hasattr(self.model, "encoder") or not hasattr(self.model.encoder, "layers"):
            return
        for block in self.model.encoder.layers:
            if hasattr(block, "attn") and hasattr(block.attn, "update_contextual_memory"):
                block.attn.update_contextual_memory(new_context)

    def adapt(self, x_new: torch.Tensor, y_new: torch.Tensor, steps: int = 1) -> float:
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
