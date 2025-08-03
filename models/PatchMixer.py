import torch
import torch.nn as nn

from layers.RevIN import RevIN


# 기존 PatchMixerLayer 그대로 사용
class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=8):
        super().__init__()

        # Depthwise Convolution + GELU + BatchNorm
        # 채널 독립적으로 Token(patch) 차원의 로컬 특징 학습
        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim,
                      kernel_size=kernel_size, # 일반적으로 Token(시계열 길이) 차원에서 Local 패턴 추출
                      groups=dim, # 채널별로 독맂벅으로 Convolution 수행. 즉, Depth-wise Convolution
                      padding='same' # 입력과 출력 길이가 동일 (Temporal Size 유지)
                      ),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )

        # Pointwise Convolution(1x1) + GELU + BatchNorm
        # 채널 간 상호작용(Channel Mixing)수행
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )

    def forward(self, x):
        x += self.Resnet(x) # Residual Connection
        x = self.Conv_1x1(x)
        return x


# PatchMixer Backbone
class PatchMixerBackbone(nn.Module):
    def __init__(self, configs, revin=True, affine=True, subtract_last=False):
        super().__init__()
        self.n_vals = configs.enc_in # 입력 변수 개수 (시계열 변수 수)
        self.lookback = configs.seq_len # 입력 시퀀스 길이
        self.forecasting = configs.pred_len # 예측 길이
        self.patch_size = configs.patch_len # 패치 크기
        self.stride = configs.stride # 패치 추출 간격
        self.kernel_size = configs.mixer_kernel_size # Depthwise Conv Kernel Size

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.a = self.patch_num
        self.d_model = configs.d_model
        self.dropout_rate = configs.head_dropout
        self.depth = configs.e_layers

        #
        '''
          Patch-Mixer Blocks 구성
          각 Block:
           - PatchMixerLayer = Depthwise Conv + GELU + BatchNorm + Pointwise Conv
           - 역할
             : Depthwise Conv -> 채널 독립적 Local Token Mixing
             : Pointwise Conv -> Channel Mixing
           - Residual 연결로 Gradient Flow 안정화  
        '''
        self.PatchMixer_blocks = nn.ModuleList([])
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(
                PatchMixerLayer(dim=self.patch_num, a=self.a, kernel_size=self.kernel_size)
            )

        self.W_P = nn.Linear(self.patch_size, self.d_model)
        self.flatten = nn.Flatten(start_dim=-2)

        self.dropout = nn.Dropout(self.dropout_rate)

        '''
            입력 정규화
                - Reversible Instance Normalization
                    : 각 시계열 채널을 개별적으로 정규화
                    : 추세나 스케일 차이를 제거하여 학습 안정성 향상
                    : 예측 후 Inverse Transformer 가능
        '''
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.n_vals, affine=affine, subtract_last=subtract_last)

    def forward(self, x):
        bs = x.shape[0]
        n_vars = x.shape[-1]

        if self.revin:
            x = self.revin_layer(x, 'norm')

        '''
            Patching
                - 입력: (Batch, 변수, 시간)
                - unfold:
                    - seq_len -> [patch_size 단위]로 분할
                    - stride 간격으로 이동
                - 결과: (batch, n_vars, patch_num, patch_size)
        '''
        x = x.permute(0, 2, 1)
        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        '''
            Linear Projection
                - 패치 길이(patch_size)를 latent dimension(d_model)로 투영
                - reshape
                    - (batch * n_vars, patch_num, d_model)
                    - 이유: 변수별로 독립 처리 (Channel Mixing 단계에서 다시 합침)
        '''
        x = self.W_P(x)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        x = self.dropout(x)

        '''
            Patch-Mixer Later 통과
                - 각 블록에서
                    - Depthwise Conv -> Token Mixing (패치 간 정보)
                    - Pointwise Conv -> Channel Mixing
                    - Residual + GELU + BatchNorm 적용
        '''
        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)

        '''
            Flatten & Aggregate
                - Flatten
                    - 패치 차원 + d_model 차원 결합 -> 글로벌 표현
                - 채널 평균
                    - Multi-variate 입력을 하나의 시계열 representation으로 집약
        '''
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

        # Projection Layer
        self.proj = nn.Sequential(
            nn.Linear(patch_repr_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        combined_dim = 128 + feature_dim

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, configs.pred_len)  # output = horizon length
        )

    def forward(self, ts_input, feature_input):
        # 1. PatchMixer Backbone 시계열 입력
        patch_repr = self.backbone(ts_input)

        # 2. Projection Layer 차원 축소
        patch_repr_proj = self.proj(patch_repr)
        # print(f"patch_repr_proj: {patch_repr_proj.shape}")

        # 3. Feature branch processing (using MLP)
        feature_repr = self.feature_mlp(feature_input)
        # print(f"feature_repr: {feature_repr.shape}")

        # 4. Concatenate patch representation layer and feature representation layer
        combined = torch.cat([patch_repr_proj, feature_repr], dim=1)
        # print(f"combined: {combined.shape}")

        # 5. Fully Connected Layers -> Final
        out = self.fc(combined)

        return out