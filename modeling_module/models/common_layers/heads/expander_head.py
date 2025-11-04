import torch.nn as nn
import torch
from modeling_module.utils.temporal_expander import TemporalExpander


class ExpanderHead(nn.Module):
    """
    z_last:[B,D] -> TemporalExpander -> [B, H, F] -> Linear(F -> 1) -> [B, H]
    """

    def __init__(
            self,
            d_model: int,
            horizon: int,
            f_out: int,
            nonneg: bool = True,
            *,
            use_sinus: bool = True,
            season_period: int = 52,
            max_harmonics: int = 16,
            use_conv: bool = True,
            dropout: float = 0.1
    ):
        super().__init__()
        self.expander = TemporalExpander(
            d_in = d_model,
            horizon = horizon,
            f_out = f_out,
            dropout = dropout,
            use_sinus = use_sinus,
            season_period=season_period,
            max_harmonics=max_harmonics,
            use_conv = use_conv,
        )

        layers = [nn.Linear(f_out, 1)]
        if nonneg:
            layers.append(nn.Softplus())
        self.proj = nn.Sequential(*layers)

    def forward(self, z_last: torch.Tensor) -> torch.Tensor:
        # z_last: [B, D]
        x_bhf = self.expander(z_last)       # [B, H, F]
        y = self.proj(x_bhf).squeeze(-1)    # [B, H]
        return y