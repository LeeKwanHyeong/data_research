import torch.nn as nn

class TrendCorrector(nn.Module):
    def __init__(self, d_model, output_horizon):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_horizon)
        )

    def forward(self, encoded_seq):
        last_token = encoded_seq[:, -1, :]
        correction = self.fc(last_token)
        return correction