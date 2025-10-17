import torch.nn as nn
import torch
from torch import Tensor
class FlattenHead(nn.Module):
    """
    Patch 기반 백본 출력을 flatten 후 horizon 크기의 예측으로 투영하는 forecasting head.

    입력:  z ∈ [B, nvars, d_model, patch_num]
    출력:  y ∈ [B, nvars, horizon]

    주요 특징:
    - 백본 출력(z)을 patch 단위 feature vector로 flatten
    - Flatten된 feature → 선형 projection → 시계열 예측
    - `individual=True` 설정 시 변수별 개별 Linear layer 사용
      예: 센서별, 상품군별 개별 패턴이 강할 때
    - `individual=False` 설정 시 공유 Linear layer 사용
      예: 모든 변수에 동일한 패턴 구조가 있다고 가정

    이 구조는 PatchTST, PatchMixer 등 patch 기반 모델의 출력과 연계하여 사용됨
    """
    def __init__(self,
                 d_model: int,
                 patch_num: int,
                 horizon: int,
                 n_vars: int,
                 *,
                 individual: bool = False,
                 dropout: float = 0.0):
        """
        Parameters:
        - d_model: 백본 출력 feature 차원 (per patch)
        - patch_num: 총 patch 개수
        - horizon: 예측 시점 수 (output dim)
        - n_vars: 입력 시계열 변수 개수
        - individual: True일 경우 변수별 개별 Linear layer 사용
        - dropout: Dropout 확률
        """
        super().__init__()
        self.individual = bool(individual)
        self.n_vars = int(n_vars)
        in_features = int(d_model * patch_num)
        out_features = int(horizon)

        if self.individual:
            # 변수별 개별 선형층: ModuleList로 각 변수에 대해 별도 Linear layer 구성
            self.proj = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, out_features, bias=True)
                )
                for _ in range(self.n_vars)
            ])
        else:
            # 전체 변수에 대해 공유 Linear layer
            self.proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, out_features, bias=True)
            )

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters:
        - z: Tensor of shape [B, n_vars, d_model, patch_num]
             백본 출력 feature map

        Returns:
        - y: Tensor of shape [B, n_vars, horizon]
             각 변수에 대한 시계열 예측 결과
        """
        if z.dim() != 4:
            raise RuntimeError(f"Flatten_Head expects (B,nvars,D,N). Got {tuple(z.shape)}")

        B, nvars, D, N = z.shape
        if nvars != self.n_vars:
            # 안전 가드: 구성과 입력 변수 수가 다르면 에러
            raise RuntimeError(f"nvars mismatch: cfg={self.n_vars}, input={nvars}")

        # Flatten: [B, nvars, D, N] → [B, nvars, N, D] → [B, nvars, N*D]
        zf = z.permute(0, 1, 3, 2).reshape(B, nvars, N * D)

        if self.individual:
            # 변수별 Linear 적용
            outs = []
            for i in range(nvars):
                yi = self.proj[i](zf[:, i, :])   # [B, horizon]
                outs.append(yi)
            y = torch.stack(outs, dim=1)         # [B, n_vars, horizon]
        else:
            # 공유 Linear 적용: proj는 [B, n_vars, in_features] → [B, n_vars, horizon]
            y = self.proj(zf)                    # [B, n_vars, horizon]

        return y