import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Flexible)
    - forward('norm'): 입력을 정규화하고, denorm에 필요한 통계를 캐시
    - forward('denorm'): 마지막 'norm' 호출 시의 통계로 복원
    - use_std=True  : (x - mean)/std  (기존 표준 RevIN; PatchMixer 기본)
      use_std=False : x - mean        (센터링 전용; 간헐수요에서 Titan 권장)
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-4,
        affine: bool = True,
        subtract_last: bool = False,
        use_std: bool = True,        # ← 추가 (기본 True라 기존 코드 영향 없음)
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        self.use_std = bool(use_std)

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        # 캐시(denorm에서 사용)
        self._cached_mean = None   # [B,1,C]
        self._cached_std  = None   # [B,1,C]
        self._cached_last = None   # [B,1,C] (subtract_last=True일 때)

    def _get_stats(self, x: torch.Tensor):
        # x: [B, L, C] or [B, H, C]
        if self.subtract_last:
            last = x[:, -1:, :]               # [B,1,C]
            self._cached_last = last.detach()
            self._cached_mean = None
            self._cached_std  = None
        else:
            mean = x.mean(dim=1, keepdim=True)                       # [B,1,C]
            var  = x.var(dim=1, unbiased=False, keepdim=True)        # [B,1,C]
            std  = torch.sqrt(var + self.eps)
            self._cached_mean = mean.detach()
            self._cached_std  = std.detach()
            self._cached_last = None

    def _apply_affine(self, x: torch.Tensor):
        if self.affine:
            w = self.affine_weight.view(1, 1, -1)
            b = self.affine_bias.view(1, 1, -1)
            return x * w + b
        return x

    def forward(self, x: torch.Tensor, mode: str):
        """
        x: [B,*,C]
        mode: {'norm','denorm'}
        """
        assert x.dim() >= 2, f"RevIN expects [B,*,C], got {x.shape}"

        if mode == 'norm':
            self._get_stats(x)
            if self.subtract_last:
                y = x - self._cached_last          # [B,*,C]
            else:
                if self.use_std:
                    y = (x - self._cached_mean) / (self._cached_std + self.eps)
                else:
                    y = x - self._cached_mean
            y = self._apply_affine(y)
            return y

        elif mode == 'denorm':
            y = x
            if self.affine:
                w = self.affine_weight.view(1, 1, -1)
                b = self.affine_bias.view(1, 1, -1)
                y = (y - b) / (w + self.eps)

            if self.subtract_last and (self._cached_last is not None):
                return y + self._cached_last
            else:
                assert (self._cached_mean is not None) and (self._cached_std is not None), \
                    "RevIN: call with mode='norm' before 'denorm'"
                if self.use_std:
                    y = y * (self._cached_std + self.eps) + self._cached_mean
                else:
                    y = y + self._cached_mean
                return y

        else:
            raise ValueError("RevIN.forward: mode must be 'norm' or 'denorm'")
