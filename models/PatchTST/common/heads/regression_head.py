import torch.nn as nn
import torch
from models.PatchTST.common.basics import SigmoidRange


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range = None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim = 1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:, :, :, -1]      # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)     # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)      # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)
        return y
