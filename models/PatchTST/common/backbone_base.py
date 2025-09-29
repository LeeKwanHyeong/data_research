import torch
import torch.nn as nn

from models.PatchTST.common.configs import PatchTSTConfig
from models.PatchTST.common.encoder import TSTiEncoder
from models.PatchTST.common.patching import compute_patch_num, do_patch
from models.common_layers.RevIN import RevIN

class BasePatchTSTBackbone(nn.Module):
    """공통: RevIN, patching, encoder 조립. Head는 서브클래스에서 구현."""
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg

        # RevIN
        self.revin = cfg.revin
        if self.revin:
            self.revin_layer = RevIN(cfg.c_in, affine=cfg.affine, subtract_last=cfg.subtract_last)

        # patching
        self.patch_num = compute_patch_num(cfg.lookback, cfg.patch_len, cfg.stride, cfg.padding_patch)

        # 백본 (Encoder)
        self.backbone = TSTiEncoder(
            c_in=cfg.c_in,
            patch_num=self.patch_num,
            patch_len=cfg.patch_len,
            cfg=cfg
        )

    # --- 서브클래스가 반드시 구현해야 할 부분 ---
    def build_head(self) -> nn.Module:
        raise NotImplementedError

    def head_forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, nvars, d_model, patch_num] -> 모드별 출력으로 변환"""
        raise NotImplementedError

    # --- 공통 forward ---
    def forward(self, x: torch.Tensor):
        # x: [B, nvars, lookback]
        if self.revin:
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)

        x_p = do_patch(x, self.cfg.patch_len, self.cfg.stride, self.cfg.padding_patch)  # [B,nvars,P,N]
        z = self.backbone(x_p)  # [B,nvars,d_model,N]

        self._last_z = z # Cash (Quantile Head에서 가져다 씀)

        y = self.head_forward(z)  # 모드별 출력

        if self.revin:
            # supervised면 [B,nvars,horizon], self-supervised면 복원형태에 따라 다름
            if y.dim()==3 and y.size(1)==self.cfg.c_in:
                y = y.permute(0,2,1)
                y = self.revin_layer(y, 'denorm')
                y = y.permute(0,2,1)

        return y