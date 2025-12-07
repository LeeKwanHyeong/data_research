from __future__ import annotations
import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    - use_std: True -> (x - mean) / (std + eps) 로 정규화하고, 복원 시 std를 곱해 스케일을 되돌립니다.
    - subtract_last: True 인 경우 마지막 시점값을 기준으로 센터링(일부 간헐수요 시 안정적)
    """
    def __init__(self, num_features: int, eps: float = 1e-5,
    affine: bool = True, subtract_last: bool = False,
    use_std: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.use_std = use_std


        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)


        # 통계 버퍼
        self.register_buffer('last', torch.zeros(1, 1, num_features))
        self.register_buffer('mean', torch.zeros(1, 1, num_features))
        self.register_buffer('std', torch.ones(1, 1, num_features))


    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            return self._norm(x)
        elif mode == 'denorm':
            return self._denorm(x)
        else:
            raise NotImplementedError(f"RevIN mode must be 'norm' or 'denorm', got {mode}")

    def _compute_stats(self, x: torch.Tensor):
        # x: [B, L, C]
        if self.subtract_last:
            self.last = x[:, -1:, :]
        else:
            # 수정: dim=(1, 0) -> dim=1 (시간 축 기준 평균)
            self.mean = x.mean(dim=1, keepdim=True).detach()

            # 수정: dim=(0, 1) -> dim=1 (시간 축 기준 분산)
        var = x.var(dim=1, keepdim=True, unbiased=False).detach()
        self.std = torch.sqrt(var + self.eps)


    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        self._compute_stats(x)
        if self.subtract_last:
            x_n = x - self.last
        else:
            x_n = x - self.mean

        if self.use_std:
            x_n = x_n / (self.std + self.eps)
        if self.affine:
            x_n = x_n * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)
        return x_n


    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.affine:
            y = (y - self.affine_bias.view(1, 1, -1)) / (self.affine_weight.view(1, 1, -1) + self.eps)

        if self.use_std:
            y = y * (self.std + self.eps)

        if self.subtract_last:
            y = y + self.last
        else:
            y = y + self.mean
        return y