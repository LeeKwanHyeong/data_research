import torch
import torch.nn as nn

from layers.RevIN import RevIN


# 기존 PatchMixerLayer 그대로 사용
class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=8):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )

    def forward(self, x):
        x += self.Resnet(x)
        x = self.Conv_1x1(x)
        return x


# PatchMixer Backbone
class PatchMixerBackbone(nn.Module):
    def __init__(self, configs, revin=True, affine=True, subtract_last=False):
        super().__init__()
        self.n_vals = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.kernel_size = configs.mixer_kernel_size

        self.PatchMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.a = self.patch_num
        self.d_model = configs.d_model
        self.dropout_rate = configs.head_dropout
        self.depth = configs.e_layers

        for _ in range(self.depth):
            self.PatchMixer_blocks.append(
                PatchMixerLayer(dim=self.patch_num, a=self.a, kernel_size=self.kernel_size)
            )

        self.W_P = nn.Linear(self.patch_size, self.d_model)
        self.flatten = nn.Flatten(start_dim=-2)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.n_vals, affine=affine, subtract_last=subtract_last)

    def forward(self, x):
        bs = x.shape[0]
        n_vars = x.shape[-1]

        if self.revin:
            x = self.revin_layer(x, 'norm')

        x = x.permute(0, 2, 1)
        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        x = self.W_P(x)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.dropout(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)

        # Global representation (flatten for merge)
        x = self.flatten(x)  # shape: (batch*n_vars, a*d_model)
        x = x.view(bs, n_vars, -1)
        x = x.mean(dim=1)  # aggregate across variables
        return x  # shape: (batch, patch_num*d_model)


# Full Model: PatchMixer + Feature Branch
class PatchMixerFeatureModel(nn.Module):
    def __init__(self, configs, feature_dim=4):
        super().__init__()
        self.backbone = PatchMixerBackbone(configs)

        # Feature Branch
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # Combined Head
        patch_repr_dim = self.backbone.a * self.backbone.d_model
        self.fc = nn.Sequential(
            nn.Linear(patch_repr_dim + 8, 64),
            nn.ReLU(),
            nn.Linear(64, configs.pred_len)  # output = horizon length
        )

    def forward(self, ts_input, feature_input):
        patch_repr = self.backbone(ts_input)
        feature_repr = self.feature_mlp(feature_input)
        combined = torch.cat([patch_repr, feature_repr], dim=1)
        out = self.fc(combined)
        return out