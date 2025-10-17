import torch
import torch.nn as nn

from modeling_module.models.PatchMixer.backbone import PatchMixerLayer
from modeling_module.models.Titan.common.memory import MemoryAttention, PositionWiseFFN


# ------------------------------------------------------------------------
# debug_tensor
# - 디버깅용 유틸 함수
# - 텐서의 shape, NaN/Inf 유무, 최댓값/최솟값 등을 출력
# ------------------------------------------------------------------------
def debug_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        print(f"[{name}] Not a tensor")
        return
    print(f"[{name}] shape: {tensor.shape}")
    print(f"  - has_nan: {torch.isnan(tensor).any().item()}, has_inf: {torch.isinf(tensor).any().item()}")
    print(f"  - max: {tensor.max().item():.4f}, min: {tensor.min().item():.4f}")


"""
TitanBackbone
├─ MemoryAttention (Contextual + Persistent + Input 기반 Self-Attn)
├─ LayerNorm + Dropout + Residual
├─ PositionWise FFN
├─ LayerNorm + Dropout + Residual
└─ → [B, L, D]
"""
# ------------------------------------------------------------------------
# TitanBackbone
# - Memory-as-Context 기반의 Attention → FFN 구조의 인코더 레이어
# - 입력: [B, L, D] → 출력: [B, L, D]
# - 내부 구성:
#     1) MemoryAttention: Persistent + Contextual memory 기반 다중 주의집중
#     2) PositionWise FFN: 시퀀스 단위 FFN
#     3) Residual + LayerNorm + Dropout
# ------------------------------------------------------------------------
class TitanBackbone(nn.Module):
    def __init__(self,
                 d_model,               # 임베딩 차원
                 n_heads,               # Multi-head 수
                 d_ff,                  # Feedforward 차원
                 contextual_mem_size,   # Contextual Memory 크기
                 persistent_mem_size,   # Persistent Memory 크기
                 dropout = 0.1):
        super().__init__()

        # MAC(Memory As Context) 기반 Attention
        self.attn = MemoryAttention(
            d_model,
            n_heads,
            contextual_mem_size,
            persistent_mem_size
        )

        # LayerNorm + Dropout + Residual
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward Layer
        self.ffn = PositionWiseFFN(
            d_model,
            d_ff,
            dropout = dropout
        )

        # LayerNorm + Dropout + Residual (FFN 후)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # x: [B, L, D]
        # ---- Attention Block ----
        attn_out = self.attn(x) # MAC Attention: [B, L, D]
        x = self.norm1(x + self.dropout(attn_out))  # Residual + Norm

        # ---- FeedForward Block ----
        ffn_out = self.ffn(x)   # FFN: [B, L, D]
        x = self.norm2(x + self.dropout(ffn_out))   # Residual + Norm
        return x    # [B, L, D]


"""
MemoryEncoder
├─ Input Projection (Linear)
├─ [TitanBackbone] × n_layers
└─ Output: [B, L, d_model]
"""
# ------------------------------------------------------------------------
# MemoryEncoder
# - Titan 모델의 인코더 블록
# - 입력 시퀀스를 Linear로 d_model로 투영 후, TitanBackbone (MAC 기반) 레이어를 반복 적용
# ------------------------------------------------------------------------
class MemoryEncoder(nn.Module):
    def __init__(self,
                 input_dim,             # 입력 차원 (ex: multivariate series의 feature 수)
                 d_model,               # Embedding dimension
                 n_layers,              # TitanBackbone 반복 개수
                 n_heads,               # Attention head 수
                 d_ff,                  # FeedForward 차원
                 contextual_mem_size,   # Contextual Memory Size
                 persistent_mem_size,   # Persistent Memory Size
                 dropout = 0.1):
        super().__init__()

        # 입력 차원 + d_model 차원으로 투영
        self.input_proj = nn.Linear(input_dim, d_model)

        # TitanBackbone Layer Stack
        self.layers = nn.ModuleList([
            TitanBackbone(d_model, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):   # x: [B, L, input_dim]
        # Embedding Projection
        x = self.input_proj(x)  # [B, L, d_model]

        # 반복적으로 TitanBackbone 통과
        for layer in self.layers:
            x = layer(x)    # [B, L, d_model]
        return x    # 최종 인코딩 시퀀스: [B, L, d_model]


# ------------------------------------------------------------------------
# PatchEmbed1D
# - 시간축을 따라 시계열을 패치화(분할)하는 모듈
# - 채널 독립 / 채널 혼합 방식 지원
# - 입력: [B, L, C] → 출력: [B, L', d_model]
# ------------------------------------------------------------------------
# Patch 임베딩: 1D Conv로 시간축 패치화 (Conv1d kernel=patch_len, stride=stride)
class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch: int, d_model: int, patch_len: int = 16, stride: int = 8,
                 channel_independent: bool = True):
        """
        Parameters:
        - in_ch: 입력 채널 수 (ex. 1 for univariate, N for multivariate)
        - d_model: 패치 임베딩 후 차원
        - patch_len: 패치 길이 (Conv1D 커널 크기)
        - stride: 패치 stride (중첩 제어)
        - channel_independent: 채널별 독립 임베딩 여부
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        if channel_independent:
            # 채널 독립 방식: 각 채널에 대해 개별적으로 depthwise conv → pointwise projection
            self.depthwise = nn.Conv1d(
                in_ch, in_ch,
                kernel_size=patch_len, stride=stride,
                groups=in_ch, bias=False  # 채널별 독립 처리
            )
            self.pointwise = nn.Conv1d(
                in_ch, d_model,
                kernel_size=1, bias=True  # 채널 통합 및 투영
            )
        else:
            # 채널 혼합 방식: 한 번에 전체 채널로부터 패치 생성
            self.mix = nn.Conv1d(in_ch, d_model, kernel_size=patch_len, stride=stride, bias=True)

    def forward(self, x):  # x: [B, L, C]
        x = x.transpose(1, 2)  # [B, C, L]
        if hasattr(self, "mix"):
            z = self.mix(x)  # [B, d_model, L']
        else:
            z = self.pointwise(self.depthwise(x))  # [B, d_model, L']
        z = z.transpose(1, 2)  # [B, L', d_model]
        return z

# ------------------------------------------------------------------------
# PatchMemoryEncoder
# - Patch → Mixer → Memory Layer 순으로 시계열을 임베딩
# - 입력: [B, L, C] → 출력: [B, L', D]
# ------------------------------------------------------------------------
class PatchMemoryEncoder(nn.Module):
    def __init__(self,
                 input_dim,             # 입력 채널 수
                 d_model,               # 모델 차원
                 n_layers,              # Encoder 레이어 수 (미사용)
                 n_heads,               # 헤드 수 (미사용)
                 d_ff,                  # Feedforward 차원
                 contextual_mem_size,   # MAC contextual memory size
                 persistent_mem_size,   # MAC persistent memory size
                 patch_len=12,          # patch length
                 patch_stride=8,        # patch stride
                 n_mixer_blocks=2,      # Mixer block 수
                 mixer_hidden=None,     # Mixer hidden layer
                 mixer_kernel=7,        # Mixer kernel size
                 dropout=0.1
                 ):
        super().__init__()
        self.patch = PatchEmbed1D(
            in_ch = input_dim,
            d_model = d_model,
            patch_len = patch_len,
            stride=patch_stride,
            channel_independent=True
        )

        # Patch Mixer Block (Residual Conv Mixer 등)
        mixer_hidden = mixer_hidden or (2 * d_model)
        self.mixers = nn.Sequential(*[
            PatchMixerLayer(d_model, mixer_hidden, kernel_size=mixer_kernel, dropout=dropout)
            for _ in range(n_mixer_blocks)
        ])

        self.layers = nn.ModuleList([
            TitanBackbone(d_model, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):  # x: [B, L, C]
        x = self.patch(x)  # [B, L′, D]
        # ★ mixers는 [B, D, L′]을 기대하도록 축 교정
        x = x.transpose(1, 2)  # [B, D, L′]
        x = self.mixers(x)  # [B, D, L′]
        x = x.transpose(1, 2)  # [B, L′, D]로 복원
        for layer in self.layers:
            x = layer(x)  # [B, L′, D]
        return x