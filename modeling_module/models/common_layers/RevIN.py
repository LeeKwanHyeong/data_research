import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps = 1e-5, affine = True, subtract_last = False):
        """
        Reversible Instance Normalization (RevIN)

        Args:
            num_features (int): 특성(feature)의 수 (마지막 차원 크기)
            eps (float): 분산 안정화를 위한 작은 값
            affine (bool): affine transform 적용 여부 (scale + bias)
            subtract_last (bool): 평균 대신 마지막 시점 값을 기준으로 정규화할지 여부
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params() # learnable scale/bias 파라미터 초기화

    def forward(self, x, mode: str):
        """
        Args:
            x (Tensor): 입력 tensor, shape = [B, L, C] 또는 [B, ..., C]
            mode (str): 'norm' 또는 'denorm' (정규화 / 역정규화)

        Returns:
            Tensor: 변환된 tensor
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # 학습 가능한 스케일(가중치)과 바이어스 초기화
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))    # shape = [C]
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))     # shape = [C]

    def _get_statistics(self, x):
        """
        평균과 표준편차 계산 (detach하여 역전파에는 영향 X)
        - 평균: [B, 1, C]
        - 표준편차: [B, 1, C]
        """
        dim2reduce = tuple(range(1, x.ndim-1))  # 시간축 등 모든 축을 평균
        if self.subtract_last:
            # 마지막 시점 값 사용 (예: forecasting에서 마지막 입력값 유지 시)
            self.last = x[:, -1, :].unsqueeze(1)    # shape = [B, 1, C]

        else:
            self.mean = torch.mean(x, dim = dim2reduce, keepdim = True).detach()

        self.stdev = torch.sqrt(torch.var(x, dim = dim2reduce, keepdim = True, unbiased = False) + self.eps).detach()

    def _normalize(self, x):
        """
        평균 또는 마지막 값을 기준으로 정규화 (instance-wise)
        """
        x = x.clone()
        if self.subtract_last:
            x -= self.last
        else:
            x -= self.mean

        x /= self.stdev # 정규화
        if self.affine:
            # affine transform 적용: x * γ + β
            x *= self.affine_weight
            x += self.affine_bias
        return x

    def _denormalize(self, x):
        """
        정규화된 입력에 대해 원래 스케일로 복원
        """
        x = x.clone()
        if self.affine:
            # affine 역변환
            x -= self.affine_bias
            x /= (self.affine_weight + self.eps * self.eps) # 안정성 확보
        x = x * self.stdev  # 원래 스케일 복원
        if self.subtract_last:
            x += self.last
        else:
            x += self.mean

        return x