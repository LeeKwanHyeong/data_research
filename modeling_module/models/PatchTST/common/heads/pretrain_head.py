import torch.nn as nn
import torch
class PretrainHead(nn.Module):
    """
    d_model 차원을 patch_len으로 projection하여 패치별 원신호를 재구성
    Input: [B, n_vars, d_model, N]
    Output: [B, n_vars, d_model, N] (unpatch에 바로 투입 가능한 형태)
    """
    def __init__(self, d_model: int, patch_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_vars, d_model, N]
        -> transpose(2, 3): [B, n_vars, N, d_model]
        -> Linear(d_model -> patch_len): [B, n_vars, N, patch_len]
        -> permute(0, 1, 3, 2): [B, n_vars, patch_len, N] (unpatch 호환)
        """
        x = x.transpose(2, 3)   # [B, n_vars, N, d_model]
        x = self.linear(self.dropout(x))    # [B, n_vars, N, patch_len]
        x = x.permute(0, 1, 3, 2)           # [B, n_vars, patch_len, N]
        return x

