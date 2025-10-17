import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()

        # 1D Convolution Layer (시간축 기준)
        self.downConv = nn.Conv1d(
            in_channels = c_in,         # 입력 채널 수 (feature 수)
            out_channels = c_in,        # 출력 채널 수 동일 (residual 목적)
            kernel_size = 3,            # 3개 시점 기준 필터
            padding = 2,                # circular padding을 고려한 설정
            padding_mode = 'circular'   # 시계열 특성을 살려 순환 패딩 적용
        )

        # Batch Normalization (채널 단위)
        self.norm = nn.BatchNorm1d(c_in)

        # 활성화 함수: ELU (Exponential Linear Unit)
        self.activation = nn.ELU()

        # MaxPooling으로 다운샘플링 (stride = 2)
        self.maxPool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x):
        """
        Args:
            x: [B, L, C] (batch, length, channels)
        Returns:
            out: [B, L_out, C]
        """
        # (B, L, C) -> (B, C, L): Conv1d는 channel-first 요구
        x = x.transpose(1, 2)

        # Conv1D -> BatchNorm -> ELU -> MaxPool
        x = self.downConv(x)    # [B, C, L]
        x = self.norm(x)        # [B, C, L]
        x = self.activation(x)  # [B, C, L]
        x = self.maxPool(x)     # [B, C, L_out]

        # 다시 (B, L_out, C)로 변환하여 반환
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff = None, dropout = 0.1, activation = 'relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # FFN 차원 (기본: d_model의 4배)
        self.attention = attention  # 전달된 attention 모듈 (MHA)

        # Position-wise Feedforward Layer (Conv1d 버전)
        self.conv1 = nn.Conv1d(in_channels = d_model, out_channels = d_ff, kernel_size = 1)
        self.conv2 = nn.Conv1d(in_channels = d_ff, out_channels = d_model, kernel_size = 1)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 활성화 함수 선택
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask = None):
        # 1. Multi-head Attention
        new_x, attn = self.attenion(
            x, x, x,
            attn_mask = attn_mask
        )

        x = x + self.dropout(new_x) # residual + dropout
        y = x = self.norm1(x)       # LayerNorm + 저장 (y: FFN 입력)

        # 2. Position-wise FeedForward (Conv1d 방식)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # [B, D, L]
        y = self.dropout(self.conv2(y).transpose(-1, 1))                    # [B, L, D]

        # 3. residual + LayerNorm
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers = None, norm_layer = None):
        """
        인코더는 여러 개의 Attention Layer(EncoderLayer)로 구성됨.
        선택적으로 ConvLayer 및 LayerNorm을 포함할 수 있음.

        Parameters:
        - attn_layers: List of EncoderLayer (필수)
        - conv_layers: List of ConvLayer (옵션, 각 EncoderLayer 뒤에 추가)
        - norm_layer: LayerNorm 등 출력 정규화 층 (옵션)
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask = None):
        """
        Forward Pass

        Parameters:
        - x: [B, L, D] 입력 시계열
        - attn_mask: attention mask (옵션)

        Returns:
        - x: [B, L, D] 인코더 최종 출력
        - attns: 각 레이어의 attention map 리스트
        """
        # x [B, L, D]
        attns = []

        # ConvLayer 있는 경우 (예: Informer에서 사용)
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask = attn_mask)  # Self-Attention
                x = conv_layer(x)                               # DownSampling via ConvLayer
                attns.append(attn)

            # 마지막 attention layer는 conv 없이 한 번 더 통과
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        # ConvLayer 없는 경우 (일반 Transformer-style)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask = attn_mask)
                attns.append(attn)
        # 마지막 출력 정규화 (있다면)
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


"""
x ---> Self-Attn --> + --> Norm --> Cross-Attn --> + --> Norm --> FFN --> + --> Norm --> out
        (masked)         |                  |                   |            |
                          -------------------                    -------------
                               Residual 1                        Residual 2 & 3
"""

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff = None, dropout = 0.1, activation = 'relu'):
        """
        Transformer-style Decoder Layer

        Parameters:
        - self_attention: 디코더 내부 자기주의(Self-Attention)
        - cross_attention: 인코더-디코더 주의(Cross-Attention)
        - d_model: 모델 차원
        - d_ff: Feedforward Network 차원 (default = 4 * d_model)
        - dropout: 드롭아웃 확률
        - activation: 'relu' 또는 'gelu'
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # 1) Self-Attention (디코더 내부 자기주의)
        self.self_attention = self_attention

        # 2) Cross-Attention (인코더 출력과 디코더 입력 연결)
        self.cross_attention = cross_attention

        # 3) Position-wise Feedforward (Conv1D 1x1)
        self.conv1 = nn.Conv1d(in_channels = d_model, out_channels = d_ff, kernel_size = 1)
        self.conv2 = nn.Conv1d(in_channels = d_ff, out_channels = d_model, kernel_size = 1)

        # 4) LayerNorms (잔차 연결 후 정규화)
        self.norm1 = nn.LayerNorm(d_model)  # self_attn 뒤
        self.norm2 = nn.LayerNorm(d_model)  # cross_attn 뒤
        self.norm3 = nn.LayerNorm(d_model)  # FFN 뒤

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask = None, cross_mask = None):
        """
        Parameters:
        - x: 디코더 입력 시퀀스 [B, L, D]
        - cross: 인코더 출력 시퀀스 [B, S, D]
        - x_mask: 디코더 자기주의용 마스크 (예: causal mask)
        - cross_mask: 인코더-디코더 주의용 마스크

        Returns:
        - output: 디코더 레이어 출력 [B, L, D]
        """

        # Self-Attention + Residual + Norm
        # self_attention(x, x, x): query, key, value 모두 디코더 입력
        x += self.dropout(self.self_attention(
            x, x, x,
            attn_mask = x_mask
        )[0])
        x = self.norm1(x)

        # Cross-Attention + Residual + Norm
        # 인코더의 출력(cross)을 key/value로 사용
        x += self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask = cross_mask
        )[0])
        y = x = self.norm2(x)

        # Feedforward Network + Residual + Norm
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) # [B, D, L]
        y = self.dropout(self.conv2(y).transpose(-1, 1))                  # [B, L, D]

        return self.norm3(x + y)


"""
input x
   ↓
[ DecoderLayer 1 ]
   ↓
[ DecoderLayer 2 ]
   ↓
  ...
   ↓
[ DecoderLayer N ]
   ↓
( optional LayerNorm )
   ↓
( optional projection )
   ↓
   output
"""
class Decoder(nn.Module):
    """
    Transformer Decoder 전체 블록

    Parameters:
    - layers: List of DecoderLayer (디코더 레이어 리스트)
    - norm_layer: 마지막 LayerNorm (선택 사항)
    - projection: 출력 차원 매핑을 위한 Linear Layer 등 (선택 사항)
    """
    def __init__(self, layers, norm_layer = None, projection = None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers) # nn.ModuleList로 래핑하여 파라미터로 등록
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask = None, cross_mask = None):
        """
        디코더 블록 전체 forward 연산

        Parameters:
        - x: 디코더 입력 [B, L, D]
        - cross: 인코더 출력 [B, S, D]
        - x_mask: 디코더용 마스크 (Self-Attn에 사용)
        - cross_mask: 인코더-디코더 마스크 (Cross-Attn에 사용)

        Returns:
        - x: 디코더 출력 [B, L, D] 또는 [B, L, output_dim] (projection 존재 시)
        """
        for layer in self.layers:
            # 각 DecoderLayer를 순차적으로 적용
            x = layer(x, cross, x_mask = x_mask, cross_mask = cross_mask)

        if self.norm is not None:
            # 전체 정규화 (예: LayerNorm)
            x = self.norm(x)

        if self.projection is not None:
            # 출력 차원 투영 (예: Linear(out_dim))
            x = self.projection(x)
        return x