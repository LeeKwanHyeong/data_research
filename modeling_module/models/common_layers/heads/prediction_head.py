import torch.nn as nn
import torch

class PredictionHead(nn.Module):
    """
    Patch 기반 백본의 출력 → forecast_len 길이의 시계열로 변환하는 예측 헤드

    입력:  x ∈ [B, nvars, d_model, num_patch]
    출력:  y ∈ [B, forecast_len, nvars]

    individual=True:
      - 변수별(ModuleList)로 독립적인 linear + dropout + flatten 사용
      - 다양한 변수별 동적 패턴을 반영하기 적합

    individual=False:
      - 모든 변수에 대해 동일한 shared linear layer 적용
      - 계산량이 적고 파라미터 공유

    flatten 옵션은 start_dim=-2를 기준으로 patch vector를 펴서 d_model * num_patch로 만들기 위함
    """
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout = 0, flatten = False):
        """
        Parameters:
        - individual: True → 변수별 Linear layer 사용
        - n_vars: 시계열 변수 수
        - d_model: 백본 embedding dim
        - num_patch: 총 patch 수
        - forecast_len: 예측 시점 수
        - head_dropout: Dropout 비율
        - flatten: 개별 flatten 연산자 사용 여부 (indiv mode일 때 사용)
        """
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * num_patch  # 각 변수별 feature vector의 차원

        if self.individual:
            # 변수별 개별 Linear + Dropout + Flatten 레이어 구성
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim = -2))        # [B, d_model, patch_num] -> [B, D*N]
                self.linears.append(nn.Linear(head_dim, forecast_len))  # 각 변수별 예측
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim = -2)                   # 전체 변수 공유 flatten
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                # i번째 변수의 patch feature 추출
                z = self.flatten[i](x[:, i, :, :])  # z: [bs x d_model * num_patch]
                z = self.linears[i](z)              # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim = 1)         # x: [bs x n_vars x forecast_len]
        else:
            # flatten 후 shared linear 적용
            x = self.flatten(x)                     # x: [bs x n_vars x (d_model * num_patch)]
            x = self.dropout(x)
            x = self.linear(x)                      # x: [bs x n_vars x forecast_len]
        return x.transpose(2, 1)        # [bs x forecast_len x n_vars]