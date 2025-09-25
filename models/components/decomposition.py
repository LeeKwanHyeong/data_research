import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CausalMovingAverage1D(nn.Module):
    """
    x: (B, C, L) -> trend: (B, C, L), residual: (B, C, L)
    kernel은 균일 평균. 과거만 보는 casual padding 사용.
    """
    def __init__(self, window: int):
        super().__init__()
        assert window > 0 and isinstance(window, int)
        # conv1d weight를 고정(평균) 커널로 생성, c 채널에 공유 적용을 윟 group conv 사용
        self.register_buffer('kernel', torch.ones(1, 1, window) / window)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (B, C, L)
        B, C, L = x.shape

        # causal pad: 오른쪽(미래) 패딩 없이, 왼쪽만 (window - 1)
        x_pad = F.pad(x, (self.window - 1, 0), mode = 'replicate')
        #group conv: 채널 별 독립 평균을 위해 groups = C
        trend = F.conv1d(x_pad, self.kernel.expand(C, -1, -1), groups = C)
        residual = x - trend
        return trend, residual

