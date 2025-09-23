import torch.nn as nn

from models.PatchTST.common.backbone_base import BasePatchTSTBackbone
from models.PatchTST.common.configs import PatchTSTConfig
from models.PatchTST.common.heads.flatten_head import Flatten_Head


class SupervisedBackbone(BasePatchTSTBackbone):
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__(cfg)
        self.head_nf = cfg.d_model * self.patch_num
        self.head = self.build_head()
        cfg = self.cfg

    def build_head(self) -> nn.Module:
        return Flatten_Head(
            individual = self.cfg.head.individual,
            n_vars = self.cfg.c_in,
            nf = self.head_nf,
            target_window = self.cfg.horizon,
            head_dropout = self.cfg.head.head_dropout
        )

    def head_forward(self, z):
        # z: [B, nvars, d_model, patch_num] -> [B, nvars, horizon]
        return self.head(z)