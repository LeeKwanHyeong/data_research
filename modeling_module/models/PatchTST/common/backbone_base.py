import torch
from torch import nn

from modeling_module.models.PatchTST.common.patching import compute_patch_num
from modeling_module.models.PatchTST.common.pos_encoding import positional_encoding
from modeling_module.models.common_layers.RevIN import RevIN


class PatchBackboneBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.c_in = cfg.c_in
        self.d_model = cfg.d_model
        self.patch_len = cfg.patch_len
        self.stride = cfg.stride
        # patch 수 계산
        from modeling_module.models.PatchTST.common.patching import compute_patch_num
        self.patch_num = compute_patch_num(cfg.lookback, cfg.patch_len, cfg.stride, cfg.padding_patch)

        # patch projection layer
        self.patch_proj = nn.Conv1d(
            in_channels=self.c_in,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            stride=self.stride,
        )

        # positional encoding
        from modeling_module.models.PatchTST.common.pos_encoding import positional_encoding
        self.pos_enc = positional_encoding(
                pe=getattr(cfg, "pe", "sincos"),          # sincos 또는 zeros
                learn_pe=getattr(cfg, "learn_pe", True),  # 학습 여부
                q_len=self.patch_num,                     # patch 개수
                d_model=self.d_model,                     # feature 차원
            )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력: x [B, C, L]
        출력: z [B, L_patch, d_model]
        """
        z = self.patch_proj(x)  # [B, d_model, L_patch]
        z = z.transpose(1, 2).contiguous()  # [B, L_patch, d_model]

        # positional encoding 직접 더하기
        z = z + self.pos_enc[: z.size(1), :].unsqueeze(0)
        return z