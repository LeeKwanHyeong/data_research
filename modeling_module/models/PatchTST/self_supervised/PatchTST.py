# modeling_module/models/PatchTST/self_supervised/PatchTST.py

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import torch
from torch import nn

from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.common_layers.RevIN import RevIN

from models.PatchTST.self_supervised.backbone import PatchTSTSelfSupBackbone


def _masked_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mse",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Args:
        pred:   [B,N,D]
        target: [B,N,D]
        mask:   [B,N] bool (True=masked positions used for loss)
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # expand mask to [B,N,1]
    m = mask.unsqueeze(-1)

    diff = pred - target
    if loss_type.lower() == "mae":
        per_elem = diff.abs()
    else:
        per_elem = diff * diff  # mse

    masked = per_elem * m  # [B,N,D]
    denom = m.sum() * pred.size(-1)  # number of masked elements
    denom = denom.clamp_min(eps)
    return masked.sum() / denom


class PatchTSTPretrainModel(nn.Module):
    """
    PatchTST Self-Supervised Pretraining Model
    - Masked patch reconstruction objective (like MAE-style but in patch space)
    - Compatible forward signature style (accepts exo args but ignores them)

    Input:
        x: [B,C,L]
    Output dict:
        {
          "loss": scalar (if return_loss=True),
          "patch_mask": [B,N] bool,
          "patch_pred": [B,C,N,patch_len],
          "patch_target": [B,C,N,patch_len],
          "z": [B,N,d_model],
        }
    """

    def __init__(self, cfg: PatchTSTConfig, attn_core=None):
        super().__init__()
        self.cfg = cfg
        self.n_vars = int(getattr(cfg, "n_vars", getattr(cfg, "enc_in", 1)))
        self.patch_len = int(getattr(cfg, "patch_len", 16))
        self.stride = int(getattr(cfg, "stride", self.patch_len))

        self.use_revin = bool(getattr(cfg, "revin", True))
        self.revin_layer = RevIN(num_features=self.n_vars) if self.use_revin else None

        self.backbone = PatchTSTSelfSupBackbone(cfg, attn_core=attn_core)

        # default pretrain settings
        self.default_mask_ratio = float(getattr(cfg, "mask_ratio", 0.3))
        self.default_loss_type = str(getattr(cfg, "pretrain_loss", "mse"))

    @classmethod
    def from_config(cls, config: PatchTSTConfig):
        return cls(cfg=config)

    def _make_patch_mask(self, B: int, N: int, mask_ratio: float, device) -> torch.Tensor:
        """
        Returns:
            mask [B,N] bool, True=masked
        """
        if mask_ratio <= 0.0:
            return torch.zeros((B, N), device=device, dtype=torch.bool)
        if mask_ratio >= 1.0:
            return torch.ones((B, N), device=device, dtype=torch.bool)

        # per-sample random masking
        rand = torch.rand((B, N), device=device)
        return rand < mask_ratio

    def forward(
        self,
        x: torch.Tensor,
        # Trainer/Adapter compatibility (ignored for self-supervised pretrain)
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids=None,
        mode: Optional[str] = None,
        # self-supervised controls
        mask_ratio: Optional[float] = None,
        patch_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        loss_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            x: [B,C,L]
            mask_ratio: float in [0,1], used if patch_mask is None
            patch_mask: optional [B,N] bool
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x as [B,C,L]. got {tuple(x.shape)}")

        B, C, L = x.shape
        if C != self.n_vars:
            # allow flexibility but keep it explicit
            raise ValueError(f"cfg.n_vars={self.n_vars} but got x.shape[1]={C}")

        # RevIN norm in time domain (standard for PatchTST family)
        if self.use_revin:
            # RevIN expects [B,L,C] in many implementations; your module may support [B,C,L].
            # Here we support both by checking.
            # We'll convert to [B,L,C] then back to [B,C,L] for safety.
            x_t = x.permute(0, 2, 1).contiguous()  # [B,L,C]
            x_n = self.revin_layer(x_t, "norm").permute(0, 2, 1).contiguous()  # [B,C,L]
        else:
            x_n = x

        # Build target patches (in normalized space)
        patches_target, N = self.backbone.patchify(x_n)  # [B,N,C*patch_len], N

        # Mask
        if patch_mask is None:
            mr = self.default_mask_ratio if mask_ratio is None else float(mask_ratio)
            patch_mask = self._make_patch_mask(B=B, N=patches_target.size(1), mask_ratio=mr, device=x.device)
        else:
            if patch_mask.dtype != torch.bool:
                patch_mask = patch_mask.bool()

        # Forward (masked reconstruction)
        z, patches_pred = self.backbone.forward_from_patches(patches=patches_target, patch_mask=patch_mask)

        # Optionally denorm in patch space is not well-defined; typically we compute loss in normalized space.
        # But we can provide a "denorm view" by reconstructing time-domain patches per-channel if needed later.

        # Reshape to [B,C,N,patch_len] for easier inspection
        patches_pred_view = patches_pred.view(B, N, C, self.patch_len).permute(0, 2, 1, 3).contiguous()
        patches_tgt_view = patches_target.view(B, N, C, self.patch_len).permute(0, 2, 1, 3).contiguous()

        out: Dict[str, Any] = {
            "patch_mask": patch_mask,                 # [B,N]
            "patch_pred": patches_pred_view,          # [B,C,N,patch_len]
            "patch_target": patches_tgt_view,         # [B,C,N,patch_len]
            "z": z,                                   # [B,N,d_model]
        }

        if return_loss:
            lt = self.default_loss_type if loss_type is None else str(loss_type)
            loss = _masked_recon_loss(
                pred=patches_pred,                    # [B,N,C*patch_len]
                target=patches_target,                # [B,N,C*patch_len]
                mask=patch_mask,                      # [B,N]
                loss_type=lt,
            )
            out["loss"] = loss

        return out

    def export_encoder_state(self) -> Dict[str, torch.Tensor]:
        """
        Downstream(지도학습 forecasting)로 encoder를 이식할 때 사용하기 위한 state dict.
        supervised 모델의 키 구조가 다를 수 있으므로, encoder/patch_embed 등 "핵심 블록"만 분리합니다.
        """
        sd = {}
        for k, v in self.state_dict().items():
            if k.startswith("backbone.patch_embed.") or k.startswith("backbone.encoder.") or k.startswith("backbone.norm_out."):
                sd[k] = v
        return sd