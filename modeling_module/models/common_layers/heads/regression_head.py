import torch.nn as nn
from modeling_module.models.PatchTST.common.basics import SigmoidRange


class RegressionHead(nn.Module):
    """
    Patch 기반 백본의 출력을 회귀값으로 변환하는 Regression Head.

    입력:
        - x: Tensor of shape [B, n_vars, d_model, num_patch]
          (예: PatchTST 또는 PatchMixer의 출력)

    출력:
        - y: Tensor of shape [B, output_dim]

    특징
    - 마지막 patch의 feature만 사용 (마지막 시점 기준)
    - flatten 후 dropout + linear projection
    - y_range 설정 시, SigmoidRange를 통해 출력값 범위 제한 가능
      (예: [0, 1] 또는 [min, max] 구간 회귀)
    """
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range = None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim = 1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x n_vars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:, :, :, -1]      # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)     # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)      # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)
        return y
