from typing import Sequence

import torch
from torch import Tensor
from torch import nn

from models.PatchTST.common.configs import PatchTSTConfig
from models.PatchTST.supervised.backbone import SupervisedBackbone
from models.components.decomposition import CausalMovingAverage1D


class BaseModel(nn.Module):
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
        pred = self.backbone(x)  # (B, nvars, H)

        # 5) 출력 차원 정규화
        pred = pred.permute(0, 2, 1).contiguous()  # (B, H, nvars)

        # 6) EXO 가산 (옵션)
        if (self.exo_head is not None) and (future_exo is not None):
            pred = pred.contiguous()
            B, Hm, C = pred.shape

            # device/dtype 정렬
            future_exo = future_exo.to(pred.device, dtype=pred.dtype)

            # 길이 정합: (B, Hm, exo_dim)로 맞추기
            if future_exo.size(1) != Hm:
                Hx = future_exo.size(1)
                exo_dim = future_exo.size(2)
                if Hx > Hm:  # 잘라냄
                    future_exo = future_exo[:, :Hm, :]
                else:  # 0 패딩
                    pad = torch.zeros(B, Hm - Hx, exo_dim, device=pred.device, dtype=pred.dtype)
                    future_exo = torch.cat([future_exo, pad], dim=1)

            # (B, Hm, 1) → broadcast add
            ex = self.exo_head(future_exo).squeeze(-1)  # (B, Hm)
            pred = pred + ex.unsqueeze(-1)  # (B, Hm, C)

        return pred

class QuantileModel(nn.Module):
    def __init__(
             self,
             configs: PatchTSTConfig,
             quantiles: Sequence[float] = (0.1, 0.5, 0.9),
             target_channel: int = 0,
             q_hidden: int = 64,
             monotonic: bool = True,
             use_exo_in_head: bool = False,
             exo_dim: int | None = None
         ):
        super().__init__(configs, exo_dim = exo_dim)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.Q = len(self.quantiles)
        self.target_channel = int(target_channel)
        self.monotonic = bool(monotonic)
        self.use_exo_in_head = bool(use_exo_in_head)

        # q_head input dimension = 1 (point scalar) + (optional) exo_dim
        in_dim = 1 + (self.exo_dim if self.use_exo_in_head and self.exo_dim > 0 else 0)

        self.q_head = nn.Sequential(
            nn.Linear(in_dim, q_hidden),
            nn.GELU(),
            nn.Linear(q_hidden, self.Q)
        )

    def _align_exo_len(self, exo: Tensor, Hm: int, device, dtype) -> Tensor:
        """
        exo: (B, H, exo_dim) -> (B, Hm, exo_dim)
        """
        B, Hx, E = exo.size()
        exo = exo.to(device = device, dtype = dtype)
        if Hx == Hm:
            return exo
        if Hx > Hm:
            return exo[:, :Hm, :]

        pad = torch.zeros(B, Hm -Hx, E, device = device, dtype = dtype)
        return torch.cat([exo, pad], dim = 1)

    def forward(self, x: Tensor, future_exo: Tensor | None = None) -> Tensor:
        """
        return: (B, Q, H) - target_channel에 대한 분위수들
        """
        # 1) Base point 예측: (B, H, C) (exo 가산은 BaseModel 내부에서 수행)
        pred_bhc = super().forward(x, future_exo) # (B, H, C)
        B, Hm, C = pred_bhc.shape

        # 2) target channel scalar만 추출: (B, H)
        if C == 1:
            pred_bh = pred_bhc.squeeze(-1) # (B, H)
        else:
            if not (0 <= self.target_channel < C):
                raise IndexError(f"target_channel = {self.target_channel} out of range [0, {C-1}]")
            pred_bh = pred_bhc[..., self.target_channel]

        # 3) q_head 입력 구성: (B, H, iin_dim)
        feats = pred_bh.unsqueeze(-1)   # (B, H, 1)
        if self.use_exo_in_head and (future_exo is not None) and (self.exo_dim > 0):
            future_exo = self._align_exo_len(
                future_exo,
                Hm,
                device = pred_bh.device,
                dtype = pred_bh.dtype
            )
            feats = torch.cat([feats, future_exo], dim = -1) # (B, H, 1 + exo_dim)

        # 4) time-distributed MLP -> (B, H, Q) -> (B, Q, H)
        q = self.q_head(feats.reshape(B * Hm, -1)).reshape(B, Hm, self.Q)   # (B, H, Q)
        q = q.permute(0, 2, 1).contiguous() # (B, Q, H)

        # 5) (optional) q10 <= q50 <= q90
        if self.monotonic:
            q, _ = torch.sort(q, dim = 1)
        return q
