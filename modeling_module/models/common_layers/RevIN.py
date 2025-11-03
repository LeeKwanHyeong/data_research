# RevIN.py
import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series
    - x: [B, L, C] 가정 (배치, 길이, 채널)
    - forward(mode='norm'): x를 정규화하고 μ,σ(또는 last)를 버퍼에 저장
    - forward(mode='denorm'): 같은 forward 컨텍스트에서 복원
    메모:
      * 본 모듈은 "한 번의 forward 호출 내"에서 norm→denorm이 이어지는 사용을 권장합니다.
      * μ,σ는 배치별·채널별 통계입니다.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)

        if self.affine:
            self._init_params()

        # forward('norm')에서 채워지고 forward('denorm')에서 사용
        self._cached_mean = None  # [B,1,C]
        self._cached_std  = None  # [B,1,C]
        self._cached_last = None  # [B,1,C]

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))  # [C]
        self.affine_bias   = nn.Parameter(torch.zeros(self.num_features)) # [C]

    def _reduce_dims(self, x: torch.Tensor):
        """
        x: [B, L, C] 혹은 [B, *, C] 라면, 시간/공간 축 전체에 대해 평균/표준편차를 구하도록
        채널 직전 모든 축을 reduce 대상으로 잡습니다.
        """
        assert x.dim() >= 3 and x.size(-1) == self.num_features, \
            f"Expected [..., C={self.num_features}] got {tuple(x.shape)}"
        # 배치/채널을 제외한 모든 축을 reduce
        dim2reduce = tuple(range(1, x.ndim - 1))
        return dim2reduce

    def _compute_stats(self, x: torch.Tensor):
        if self.subtract_last:
            # 마지막 시점(축=-2)의 값을 빼는 설정
            last = x.select(dim=-2, index=x.size(-2) - 1).unsqueeze(-2)  # [B,1,C]
            mean = None
            std  = None
        else:
            dim2reduce = self._reduce_dims(x)
            mean = x.mean(dim=dim2reduce, keepdim=True)  # [B,1,C]
            var  = x.var(dim=dim2reduce, keepdim=True, unbiased=False)
            std  = torch.sqrt(var + self.eps)            # [B,1,C]
            last = None
        return mean, std, last

    def _apply_affine(self, x: torch.Tensor):
        if not self.affine:
            return x
        w = self.affine_weight.view(1, *([1] * (x.ndim - 2)), -1)  # [..., C]
        b = self.affine_bias.view( 1, *([1] * (x.ndim - 2)), -1)
        return x * w + b

    def _inverse_affine(self, x: torch.Tensor):
        if not self.affine:
            return x
        w = self.affine_weight.view(1, *([1] * (x.ndim - 2)), -1)
        b = self.affine_bias.view( 1, *([1] * (x.ndim - 2)), -1)
        return (x - b) / (w + 1e-12)

    def forward(self, x: torch.Tensor, mode: str):
        """
        mode in {'norm', 'denorm'}
        - 'norm': x[... , C] 기준으로 μ,σ(혹은 last)를 구해 정규화. 캐시에 저장.
        - 'denorm': 직전 'norm'에서 계산된 캐시로 복원.
        """
        if mode == 'norm':
            assert x.size(-1) == self.num_features, \
                f"RevIN expected last dim C={self.num_features}, got {x.size(-1)}"
            mean, std, last = self._compute_stats(x)
            self._cached_mean = mean
            self._cached_std  = std
            self._cached_last = last

            if self.subtract_last:
                x_n = x - last  # [B,*,C]
                x_n = self._apply_affine(x_n)
            else:
                x_n = (x - mean) / std
                x_n = self._apply_affine(x_n)

            return x_n

        elif mode == 'denorm':
            # 입력 y는 정규화 공간 값. 캐시가 존재해야 함.
            if self.subtract_last:
                assert self._cached_last is not None, "RevIN denorm: last cache missing. Call norm first in same forward."
                y = self._inverse_affine(x)
                y = y + self._cached_last
            else:
                assert (self._cached_mean is not None) and (self._cached_std is not None), \
                    "RevIN denorm: mean/std cache missing. Call norm first in same forward."
                y = self._inverse_affine(x)
                y = y * self._cached_std + self._cached_mean
            return y

        else:
            raise ValueError("mode must be 'norm' or 'denorm'")

    # ----- Utilities for model outputs (H may differ from L) -----
    @torch.no_grad()
    def denorm_like_channel(self, out: torch.Tensor, target_channel: int = 0) -> torch.Tensor:
        """
        모델 출력(out)이 마지막 차원이 C가 아닐 수 있는 경우(예: [B,H] 또는 [B,H,Q])에
        '타깃 채널'의 μ,σ (혹은 last)만 사용해 스칼라/벡터에 동일 스케일을 곱해 복원.
        - out: [B,H] or [B,H,1] or [B,H,Q] ... (C와 무관한 텐서)
        - 반환: 동일 shape, raw 공간
        """
        B = out.size(0)
        if self.subtract_last:
            assert self._cached_last is not None
            last = self._cached_last[:, :, target_channel:target_channel+1]  # [B,1,1]
            # out의 shape에 맞게 broadcast
            while last.dim() < out.dim():
                last = last.expand(-1, -1, *([-1] * (last.dim() - 2)))
            # [B,1,1] -> [B,H,1] 혹은 [B,H,Q]로 broadcast
            last = last.expand(B, *( [out.size(d) for d in range(1, out.dim())] ))
            y = out + last
        else:
            assert (self._cached_mean is not None) and (self._cached_std is not None)
            mean = self._cached_mean[:, :, target_channel:target_channel+1]  # [B,1,1]
            std  = self._cached_std[:,  :, target_channel:target_channel+1]  # [B,1,1]
            mean = mean.expand(B, *( [out.size(d) for d in range(1, out.dim())] ))
            std  = std.expand( B, *( [out.size(d) for d in range(1, out.dim())] ))
            y = out * std + mean
        return y
