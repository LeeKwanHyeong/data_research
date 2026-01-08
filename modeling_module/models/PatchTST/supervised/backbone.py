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
        # 1. Attention Core 자동 선택
        # -----------------------------------
        if attn_core is None:
            attn_type = getattr(cfg.attn, "type", "full").lower()

            if attn_type == "full":
                attn_core = FullAttention(
                    mask_flag=cfg.attn.causal,
                    attention_dropout=cfg.attn.attn_dropout,
                    output_attention=cfg.attn.output_attention,
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
        # 2. Encoder Layer Stack
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
    # 3. Forward (Patchify → Encoder)
    # -----------------------------------
    def forward(self, x: torch.Tensor, p_cont: torch.Tensor = None, p_cat: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Target [B, L, C] (Revin Normalized)
            p_cont: Past Exo Continuous [B, L, d_past_cont]
            p_cat: Past Exo Categorical [B, L, d_past_cat]
        Returns:
            z: [B, N, d_model]
        """
        # 1) Patchify & Embed (with Exo)
        z = self._patchify_and_embed(x, p_cont, p_cat)  # [B, N, d_model]

        # 2) Transformer Encoder
        z = self.encoder(z)  # [B, N, d_model]

        # 3) Output Norm
        z = self.norm_out(z)

        return z