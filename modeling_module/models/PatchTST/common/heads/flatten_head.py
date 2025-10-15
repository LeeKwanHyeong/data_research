import torch.nn as nn
import torch
import torch.nn as nn
from torch import Tensor
class Flatten_Head(nn.Module):
    """
    입력:  z [B, nvars, d_model, patch_num]
    출력:  y [B, nvars, horizon]

    - individual=False: 모든 변수에 공유 선형층 (D*N -> H)
    - individual=True : 변수별로 다른 선형층 (ModuleList)
    """
    def __init__(self,
                 d_model: int,
                 patch_num: int,
                 horizon: int,
                 n_vars: int,
                 *,
                 individual: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.individual = bool(individual)
        self.n_vars = int(n_vars)
        in_features = int(d_model * patch_num)
        out_features = int(horizon)

        if self.individual:
            self.proj = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, out_features, bias=True)
                )
                for _ in range(self.n_vars)
            ])
        else:
            self.proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, out_features, bias=True)
            )

    def forward(self, z: Tensor) -> Tensor:
        """
        z: [B, nvars, d_model, patch_num]
        return: [B, nvars, horizon]
        """
        if z.dim() != 4:
            raise RuntimeError(f"Flatten_Head expects (B,nvars,D,N). Got {tuple(z.shape)}")

        B, nvars, D, N = z.shape
        if nvars != self.n_vars:
            # 안전 가드: 구성과 입력 변수 수가 다르면 에러
            raise RuntimeError(f"nvars mismatch: cfg={self.n_vars}, input={nvars}")

        # [B, nvars, D, N] -> [B, nvars, N, D] -> [B, nvars, N*D]
        zf = z.permute(0, 1, 3, 2).reshape(B, nvars, N * D)

        if self.individual:
            outs = []
            for i in range(nvars):
                yi = self.proj[i](zf[:, i, :])   # [B, H]
                outs.append(yi)
            y = torch.stack(outs, dim=1)         # [B, nvars, H]
        else:
            # 공유 프로젝션을 변수 차원에 독립적으로 적용
            y = self.proj(zf)                    # [B, nvars, H]

        return y