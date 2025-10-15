import torch.nn as nn
import torch

class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout = 0, flatten = False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim = -2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim = -2)
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
                z = self.flatten[i](x[:, i, :, :])  # z: [bs x d_model * num_patch]
                z = self.linears[i](z)              # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim = 1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)                     # x: [bs x nvars x (d_model * num_patch)]
            x = self.dropout(x)
            x = self.linear(x)                      # x: [bs x nvars x forecast_len]
        return x.transpose(2, 1)        # [bs x forecast_len x nvars]