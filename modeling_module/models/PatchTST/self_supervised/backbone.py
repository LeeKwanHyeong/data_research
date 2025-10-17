import torch
import torch.nn as nn

from modeling_module.models.PatchTST.common.backbone_base import BasePatchTSTBackbone
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.common_layers.heads.pretrain_head import PretrainHead


class SelfSupervisedBackbone(BasePatchTSTBackbone):
    def __init__(self, cfg: PatchTSTConfig):
        # Calution: SSL PretrainHead는 [B, n_vars, d_model, N] -> [B, n_vars, patch_len, N] 형태를 사용하므로
        # head_nf는 실제로 쓰지 않지만 관성적으로 보관
        super().__init__(cfg)
        self.head_nf = cfg.d_model * self.patch_num
        self.head = self.build_head()

    def build_head(self) -> nn.Module:
        return PretrainHead(
            d_model = self.cfg.d_model,
            patch_len = self.cfg.patch_len,
            dropout = self.cfg.head.head_dropout
        )

    def head_forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, n_vars, d_model, N]
        return: [B, n_vars, patch_len, N] # unpatch로 lookback 복원 가능
        """
        out = self.head(z)
        return out