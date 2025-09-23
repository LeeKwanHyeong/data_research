from typing import Optional

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from model_runner.model_configs import PatchTSTConfig


class Model(nn.Module):
    def __init__(self, configs = PatchTSTConfig):
        super().__init__()
        self.configs = configs

        # load parameters
        c_in = self.configs.c_in
        lookback = self.configs.lookback
        horizon = self.configs.horizon

        n_layers = self.configs.n_layers
        n_heads = self.configs.n_heads
        d_model = self.configs.d_model
        d_ff = self.configs.d_ff
        dropout = self.configs.dropout
        fc_dropout = self.configs.fc_dropout
        head_dropout = self.configs.head_dropout

        individual = self.configs.individual

        patch_len = self.configs.patch_len
        stride = self.configs.stride
        padding_patch = self.configs.padding_patch

        revin = self.configs.revin
        affine = self.configs.affine
        subtract_last = self.configs.subtract_last

        decomposition = self.configs.decomposition
        kernel_size = self.configs.kernel_size

        # model
        if decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=verbose, **kwargs
            )
            self.model_res = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=verbose, **kwargs
            )
        else:
            self.model = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
               patch_len=patch_len, stride=stride,
               max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
               n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
               attn_dropout=attn_dropout,
               dropout=dropout, act=act, key_padding_mask=key_padding_mask,
               padding_var=padding_var,
               attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
               store_attn=store_attn,
               pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
               padding_patch=padding_patch,
               pretrain_head=pretrain_head, head_type=head_type, individual=individual,
               revin=revin, affine=affine,
               subtract_last=subtract_last, verbose=verbose, **kwargs
            )

    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2,
                                                                                 1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x