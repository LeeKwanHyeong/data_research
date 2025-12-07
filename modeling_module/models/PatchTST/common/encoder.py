from typing import Optional

import torch
from torch import nn

from modeling_module.models.common_layers.Attention import build_attention
from modeling_module.models.PatchTST.common.basics import Transpose, get_activation_fn
from modeling_module.models.PatchTST.common.pos_encoding import positional_encoding


class TSTEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff = 256,
                 norm = 'BatchNorm',
                 dropout = 0.0,
                 activation = 'gelu',
                 mha: nn.Module = None,
                 pre_norm = False,
                 store_attn = False,
                 use_residual_logits = True):
        super().__init__()
        assert mha is not None, "mha 모듈 필요. (build_attention으로 생성하여 주입)."
        self.mha = mha
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.use_residual_logits = use_residual_logits

        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = (nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2)
        ) if 'batch' in norm.lower() else nn.LayerNorm(d_model))

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = (nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2)
        ) if 'batch' in norm.lower() else nn.LayerNorm(d_model))

    def forward(self, src, prev_logits=None, attn_mask=None):
        # src: [B,L,d_model]
        # 수정: self.norm1 -> self.norm_attn
        x = self.norm_attn(src) if self.pre_norm else src

        out, attn, logits = self.mha(x, attn_mask=attn_mask, prev_logits=prev_logits)
        if self.store_attn: self.attn = attn

        src = src + self.dropout_attn(out)

        # 수정: self.norm2 -> self.norm_ffn
        y = self.norm_ffn(src) if self.pre_norm else src
        y = self.ff(y)
        out = src + y
        return out, logits

class TSTEncoder(nn.Module):
    def __init__(self, q_len, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()

        for _ in range(cfg.n_layers):
            cfg.attn.d_model = cfg.d_model
            mha = build_attention(cfg.attn)
            self.layers.append(
                TSTEncoderLayer(
                    d_model = cfg.d_model,
                    d_ff = cfg.d_ff,
                    norm = cfg.norm,
                    dropout = cfg.dropout,
                    activation = cfg.act,
                    mha = mha,
                    pre_norm = cfg.pre_norm,
                    store_attn = cfg.store_attn,
                    use_residual_logits = cfg.attn.residual_logits
                )
            )

    def forward(self, src: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        logits = None
        x = src
        for layer in self.layers:
            x, logits = layer(x, prev_logits = logits, attn_mask = attn_mask)
        return x

class TSTiEncoder(nn.Module): # channel-independent
    def __init__(self, c_in, patch_num, patch_len, cfg):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.cfg = cfg

        q_len = patch_num
        self.W_P = nn.Linear(patch_len, cfg.d_model)    # (P -> d_model)
        self.seq_len = q_len
        self.W_pos = positional_encoding(cfg.pe, cfg.learn_pe, q_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = TSTEncoder(q_len, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, nvars, patch_len, patch_num]
        B, n_vars, _, _ = x.shape
        x = x.permute(0, 1, 3, 2)           # [B, n_vars, N, P]
        x = self.W_P(x)                     # [B, n_vars, N, d_model]

        u = x.reshape(B * n_vars, x.size(2), x.size(3)) # [B * n_vars, N, d_model]
        u = self.dropout(u + self.W_pos)                # PE 추가

        z = self.encoder(u)                             # [B * n_vars, N, d_model]
        z = z.reshape(B, n_vars, x.size(2), z.size(2))  # [B, n_vars, N, d_model]
        z = z.permute(0, 1, 3, 2)                       # [B, n_vars, d_model, N]
        return z
