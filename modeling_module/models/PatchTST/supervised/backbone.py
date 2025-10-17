import torch
from torch import nn

from modeling_module.models.PatchTST.common.backbone_base import PatchBackboneBase
from modeling_module.models.PatchTST.common.encoder import TSTEncoder, TSTEncoderLayer
from modeling_module.models.common_layers.Attention import MultiHeadAttention, FullAttention, ProbAttention, \
    FullAttentionWithLogits


class SupervisedBackbone(PatchBackboneBase):
    """
    PatchTST Supervised Backbone
    - 입력: (B, C, L)
    - 출력: (B, L_tok, d_model)
    - lookback, patch_len, stride가 동적으로 바뀌어도 안정 작동
    - attn_core=None이면 자동으로 Full / ProbSparse Attention 선택
    """

    def __init__(self, cfg, attn_core=None):
        super().__init__(cfg)
        self.cfg = cfg

        # -----------------------------------
        # [1] Attention Core 자동 선택
        # -----------------------------------
        if attn_core is None:
            attn_type = getattr(cfg.attn, "type", "full").lower()

            if attn_type == "full":
                attn_core = FullAttention(
                    mask_flag=cfg.attn.causal,
                    attention_dropout=cfg.attn.attn_dropout,
                    output_attention=cfg.attn.output_attention,
                    # residual=cfg.attn.residual_logits,
                )
            elif attn_type == "probsparse":
                attn_core = ProbAttention(
                    mask_flag=cfg.attn.causal,
                    factor=cfg.attn.factor,
                    attention_dropout=cfg.attn.attn_dropout,
                    output_attention=cfg.attn.output_attention,
                    # residual=cfg.attn.residual_logits,
                )
            elif attn_type == 'full_logits':
                attn_core = FullAttentionWithLogits(
                    mask_flag=True,
                    attention_dropout=0.1,
                    output_attention=False,
                    return_logits=True,
                    use_realformer_residual=True
                )
            else:
                raise ValueError(f"Invalid attention type: {attn_type}")
        self.attn_core = attn_core

        # -----------------------------------
        # [2] Encoder Layer Stack
        # -----------------------------------
        layers = []
        for i in range(cfg.n_layers):
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
                    mha=mha,
                    d_model=self.d_model,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    pre_norm=cfg.pre_norm,
                    store_attn=getattr(cfg, "store_attn", False),
                )
            )

        self.encoder = TSTEncoder(q_len=self.patch_num, cfg=cfg)
        self.norm_out = nn.LayerNorm(cfg.d_model)

    # -----------------------------------
    # [3] Forward (Patchify → Encoder)
    # -----------------------------------
    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        """
        입력 x_bcl: [B, C, L]
        반환 z: [B, L_tok, d_model]
        """
        # Patch embedding + positional encoding
        z = self._patchify(x_bcl)           # [B, L_tok, d_model]

        # Transformer encoder 통과
        z = self.encoder(z)                 # [B, L_tok, d_model]

        # Output 정규화
        z = self.norm_out(z)
        return z