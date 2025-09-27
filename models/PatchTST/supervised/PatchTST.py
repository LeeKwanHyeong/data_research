import torch
from torch import Tensor
from torch import nn

from models.PatchTST.common.configs import PatchTSTConfig
from models.PatchTST.supervised.backbone import SupervisedBackbone
from models.components.decomposition import CausalMovingAverage1D


class Model(nn.Module):
    """
    입력:  x (B, L, C)  - Batch, Lookback, Channels
    출력:  ŷ (B, H, C)  - Batch, Horizon, Channels (외부 파이프라인 정합을 위해 통일)
    """
    def __init__(self, configs: PatchTSTConfig, exo_dim: int | None = None):
        super().__init__()
        self.configs = configs

        # ----- PatchTST Backbone -----
        # 백본은 (B, C, L) 입력을 받아 Flatten_Head로 (B, nvars, H) 반환
        self.backbone = SupervisedBackbone(self.configs)

        # ----- Decomposition 옵션 -----
        self.decomp_type = (configs.decomp.type or 'none').lower()
        if self.decomp_type == 'ma':
            self.decomp = CausalMovingAverage1D(window=configs.decomp.ma_window)
        else:
            self.decomp = None

        self.concat_mode = configs.decomp.concat_mode  # 'concat' | 'residual_only' | 'trend_only'

        # concat이면 입력 채널이 2C가 되므로 1x1 Conv로 2C→C 매핑 (백본 c_in과 정합 유지)
        c_in = configs.c_in
        if self.decomp and self.concat_mode == 'concat':
            self.input_mapper = nn.Conv1d(in_channels=2 * c_in, out_channels=c_in, kernel_size=1, bias=False)
        else:
            self.input_mapper = nn.Identity()

        # ----- Future EXO ------
        # 우선 config.exo_dim을 우선, 인자로 덮어쓰기
        self.exo_dim = int(getattr(configs, 'exo_dim', 0) if exo_dim is None else exo_dim)
        self.horizon = configs.horizon

        # exo_dim > 0이면 (B, H, exo_dim) -> (B, H, 1) time-distributed 가산항
        self.exo_head = None
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)    # (B, H, exo_dim) -> (B, H, 1)
            )

    def _apply_decomposition(self, x_bcl: Tensor) -> Tensor:
        """x_bcl: (B, C, L) -> (B, C or 2C, L)"""
        if not self.decomp:
            return x_bcl
        trend, residual = self.decomp(x_bcl)  # (B, C, L), (B, C, L)
        if self.concat_mode == 'concat':
            x_bcl = torch.cat([trend, residual], dim=1)  # (B, 2C, L)
        elif self.concat_mode == 'trend_only':
            x_bcl = trend
        else:  # 'residual_only'
            x_bcl = residual
        return x_bcl

    def forward(self, x: Tensor, future_exo: Tensor | None = None) -> Tensor:
        """
        x: (B, L, C)  -> permute -> (B, C, L) -> [decomp] -> input_mapper -> backbone
        backbone 출력: (B, nvars, H) -> permute -> (B, H, nvars)
        """
        # 1) 입력 차원 정렬
        x = x.permute(0, 2, 1)  # (B, C, L)

        # 2) 분해(옵션)
        if self.decomp:
            x = self._apply_decomposition(x)

        # 3) 채널 정합 (concat일 때 2C→C)
        x = self.input_mapper(x)

        # 4) 백본 통과
        y = self.backbone(x)  # (B, nvars, H)

        # 5) 출력 차원 정규화
        y = y.permute(0, 2, 1)  # (B, H, nvars)

        # 6) EXO 가산 - 모든 변수에 같은 시간별 shift 덧셈
        if (self.exo_head is not None) and (future_exo is not None):
            # future_exo: (B, H, exo_dim) -> (B, H, 1) -> (B, H, 1) broadcast
            ex = self.exo_head(future_exo).squeeze(-1)  # (B, H)
            y = y + ex.unsqueeze(-1)                    # (B, H, 1) + (B, H, C)

        return y