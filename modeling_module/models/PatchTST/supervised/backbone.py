import torch
from torch import nn

from modeling_module.models.PatchTST.common.backbone_base import PatchBackboneBase
from modeling_module.models.PatchTST.common.encoder import TSTEncoder, TSTEncoderLayer
from modeling_module.models.PatchTST.common.attention import MultiHeadAttention


class SupervisedBackbone(PatchBackboneBase):
    """
    PatchTST Supervised Backbone
    - 입력: (B, C, L)
    - 출력: (B, L_tok, d_model)
    """
    def __init__(self, cfg, attn_core):
        super().__init__(cfg)
        self.cfg = cfg
        self.attn_core = attn_core
        self.d_ff = cfg.d_ff
        self.dropout = cfg.dropout
        self.pre_norm = cfg.pre_norm
        self.store_attn = getattr(cfg, "store_attn", False)

        # ---- Encoder 구성 ----
        layers = []
        for _ in range(cfg.n_layers):
            mha = MultiHeadAttention(
                d_model=self.d_model,
                n_heads=cfg.attn.n_heads,
                d_k=cfg.attn.d_k,
                d_v=cfg.attn.d_v,
                proj_dropout=cfg.attn.proj_dropout,
                attn_core=self.attn_core,
            )
            layers.append(
                TSTEncoderLayer(
                    mha,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    pre_norm=self.pre_norm,
                    store_attn=self.store_attn,
                )
            )
        self.encoder = TSTEncoder(layers, d_model=self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        """
        x_bcl: [B, C, L]
        return: [B, L_tok, d_model]
        """
        # 1) Patchify 입력
        z = self._patchify(x_bcl)               # [B, L_tok, d_model]

        # 2) Transformer Encoder
        z = self.encoder(z)                     # [B, L_tok, d_model]

        # 3) 출력 정규화
        z = self.norm_out(z)
        return z