import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CasualMovingAverage1D(nn.Module):
    """
    x: (B, C, L) -> trend: (B, C, L), residual: (B, C, L)
    kernel은 균일 평균. 과거만 보는 casual padding 사용.
    """
