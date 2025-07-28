import torch
import torch.nn as nn
import torch.nn.functional as F

# Mixer Block
class MixerBlock(nn.Module):
    def __init__(self, num_tokens, num_channels, hidden_dim):
        super(MixerBlock, self).__init__()

        # Token Mixing
        self.token_norm = nn.LayerNorm(num_channels)
        self.token_fc1 = nn.Linear(num_tokens, hidden_dim)
        self.token_fc2 = nn.Linear(hidden_dim, num_tokens)

        # Channel Mixing
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_fc1 = nn.Linear(num_channels, hidden_dim)
        self.channel_fc2 = nn.Linear(hidden_dim, num_channels)

    def forward(self, x):
        # x: (batch, tokens, channels)
        y = self.token_norm(x)
        y = y.transpose(1, 2) # batch channels tokens
        y = F.relu(self.token_fc1(y))
        y = self.token_fc2(y)
        y = y.transpose(1, 2)
        x = x + y # Residual

        z = self.channel_norm(x)
        z = F.relu(self.channel_fc1(z))
        z = self.channel_fc2(z)
        return x + z # Residual

# Mixer + Feature Branch Model
class MixerFeatureModel(nn.Module):
    def __init__(self, num_tokens = 52, num_channels = 1, mixer_hidden = 64, feature_dim = 4):
        super(MixerFeatureModel, self).__init__()

        # Mixer Branch
        self.mixer_blocks = nn.ModuleList([MixerBlock(num_tokens, num_channels, mixer_hidden) for _ in range(4)])
        self.global_pooling = nn.AdaptiveAvgPool1d(1) # For token dimension Pooling

        # Feature Branch
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # Final Combination
        self.fc = nn.Sequential(
            nn.Linear(num_channels + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output: next-step or specific horizon forecast
        )

    def forward(self, ts_input, feature_input):
        # ts_input: (batch, num_tokens, num_channels)
        x = ts_input
        for block in self.mixer_blocks:
            x = block(x)

        # Global pooling along tokens (convert to (batch, num_channels))
        x = x.transpose(1, 2) # (batch, num_channels, tokens)
        x = self.global_poolling(x).squeeze(-1)

        # Feature branch
        y = self.feature_mlp(feature_input)

        # Merge
        combined = torch.cat([x, y], dim= 1)
        out = self.fc(combined)
        return out