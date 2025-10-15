import torch
from torch import nn

from modeling_module.models.PatchTST.common.patching import compute_patch_num
from modeling_module.models.PatchTST.common.pos_encoding import positional_encoding


class PatchBackboneBase(nn.Module):
    """
    PatchTST backbone의 공통 기반 클래스.
    - 입력: (B, C, L)
    - 출력: (B, L_tok, d_model)
    - patch_len, stride 변경 시에도 안전하게 patch_num 계산
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.c_in = cfg.c_in
        self.d_model = cfg.d_model
        self.patch_len = cfg.patch_len
        self.stride = cfg.stride
        self.padding_patch = getattr(cfg, "padding_patch", None)
        self.use_pos_enc = getattr(cfg, "use_pos_enc", True)

        # Patch projection: (B, C, L) -> (B, d_model, L_tok)
        self.patch_proj = nn.Conv1d(
            in_channels=self.c_in,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            stride=self.stride,
        )

        # Positional Encoding
        self.pos_enc = positional_encoding(self.d_model)

    def _patchify(self, x_bcl: torch.Tensor) -> torch.Tensor:
        """
        x_bcl: [B, C, L_in]
        return: [B, L_tok, d_model]
        """
        B, C, L_in = x_bcl.shape

        # 안전 patch 수 계산
        L_tok = compute_patch_num(L_in, self.patch_len, self.stride, self.padding_patch)
        z = self.patch_proj(x_bcl)              # [B, d_model, L_tok]
        z = z.transpose(1, 2).contiguous()      # [B, L_tok, d_model]

        if self.use_pos_enc:
            z = z + self.pos_enc[:, :L_tok, :].to(z.device)

        return z