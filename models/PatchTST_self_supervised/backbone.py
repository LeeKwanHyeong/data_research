from typing import Optional

import torch.nn as nn
from torch import Tensor
import torch

from models.PatchTST_self_supervised.basics import Transpose
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer, FullAttentionWithLogits, \
    AttentionLayerWithPrev


# -------------------------
# PatchTST_self_supervised Backbone
# -------------------------
class PatchTSTEncoder(nn.Module):
    def __init__(self,
                 c_in,
                 num_patch,
                 patch_len,
                 n_layers = 3,
                 d_model = 128,
                 n_heads = 16,
                 shared_embedding = True,
                 d_ff = 256,
                 norm = 'BatchNorm',
                 attn_dropout = 0.,
                 dropout = 0.,
                 act = 'gelu',
                 store_attn = False,
                 res_attention = True,
                 pre_norm = False,
                 pe = 'zeros',
                 learn_pe = True,
                 verbose = False,
                 **kwargs
                 ):
        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # Positional Encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual Dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            d_model, n_heads, d_ff = d_ff, norm = norm, attn_dropout = attn_dropout,
            dropout = dropout, pre_norm = pre_norm, activation = act, res_attention = res_attention,
            n_layers = n_layers, store_attn = store_attn
        )

    def forward(self, x) -> Tensor:
        """
        x: tensor [bs x num_patch x n_vars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, :, i, :])
                x_out.append(z)
            x = torch.stack(x_out, dim = 2)
        else:
            x = self.W_P(x)                 # x: [bs x num_patch x n_vars x d_model]
        x = x.transpose(1, 2)   # x: [bs x n_vars x num_patch x d_model]

        u = torch.reshape(
            x, (bs * n_vars, num_patch, self.num_patch, self.d_model)
        )                                   # u: [bs * n_vars x num_patch x d_model]
        u = self.dropout(u + self.W_pos)    # u: [bs * n_vars x num_patch x d_model]

        # Encoder
        z = self.encoder(u)                 # z: [bs * n_vars x num_patch x d_model]
        z = torch.reshape(
            z, (-1, n_vars, num_patch, self.d_model)
        )                                   # z: [bs x n_vars x num_patch x d_model]
        z = z.permute(0, 1, 3, 2)           # z: [bs x n_vars x d_model x num_patch]

        return z

class TSTEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 d_ff = None,
                 norm = 'BatchNorm',
                 attn_dropout = 0.,
                 dropout = 0.,
                 activation = 'gelu',
                 res_attention = False,
                 n_layers = 1,
                 pre_norm = False,
                 store_attn = False):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(d_model, n_heads = n_heads, d_ff = d_ff,
                           norm = norm, attn_dropout = attn_dropout,
                           dropout = dropout, activation = activation,
                           res_attention = res_attention, pre_norm = pre_norm,
                           store_attn = store_attn) for i in range(n_layers)
        ])

        self.res_attention = res_attention

    def forward(self, src: Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev = scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 d_ff = 256,
                 store_attn = False,
                 norm = 'BatchNorm',
                 attn_dropout = 0.,
                 dropout = 0.,
                 bias = True,
                 activation = 'gelu',
                 res_attention = False,
                 causal = True, # causal mask 추가 여부
                 pre_norm = False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.res_attention = res_attention
        assert not res_attention, "res_attention is False"


        # ----- Default MHA -----
        if not res_attention:
            self.self_attn = AttentionLayer(
                attention = FullAttention(
                    mask_flag = causal,
                    attention_dropout = attn_dropout,
                    output_attention = True,    # attn 가중치 변환
                ),
                d_model = d_model,
                n_heads = n_heads,
            )
        # ----- RealFormer logits 잔차 포함 MHA -----
        else:
            attn_core = FullAttentionWithLogits(
                mask_flag = causal,
                attention_dropout = attn_dropout,
                output_attention = True,
                return_logits = res_attention,              # logits 반환 필요 시
                use_realformer_residual = res_attention,    # prev_logits 더하기
            )
            self.self_attn = AttentionLayerWithPrev(
                attention = attn_core,
                d_model = d_model,
                n_heads = n_heads,
            )

        # ---------------------
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if 'batch' in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # FFN
        self.dropout_ffn = nn.Dropout(dropout)
        if 'batch' in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: torch.Tensor, prev: Optional[torch.Tensor] = None):
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)

        # (Q, K, V 모두 src: self-attention)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, attn_mask = None, prev_logits = prev)
        else:
            src2, attn = self.self_attn(src, src, src, attn_mask = None)
            scores = None

        if self.store_attn:
            self.attn = attn    # (B, H, L, S)

        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        # FFN sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)

        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        src = src + self.dropout_ffn(src2)

        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores # scores: (B, H, L, S) = 다음 레이어 prev로 전달
        else:
            return src