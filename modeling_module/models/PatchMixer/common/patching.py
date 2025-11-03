import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_module.models.PatchMixer.common.configs import PatchMixerConfigMonthly


class SimpleUnfoldProjector(nn.Module):
    """
    (patch_len, stride)로 unfold -> W_P Projection
    input: x (B, L, N)
    output: (B * N, A, D)
    """
    def __init__(self, cfg: PatchMixerConfigMonthly, d_model: int):
        super().__init__()
        self.P: int = int(cfg.patch_len)
        self.S: int = int(cfg.stride)
        base: int = int((cfg.lookback - self.P) / self.S + 1)
        self.patch_num: int = base + 1
        self.d_model: int = int(d_model)

        self.W_P = nn.Linear(self.P, self.d_model)
        self.pad = nn.ReplicationPad1d((0, self.S))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N)
        B, L, N = x.shape
        x = x.permute(0, 2, 1)                      # (B, N, L)
        x = self.pad(x).unfold(-1, self.P, self.S)  # (B, N, A, P)
        x = self.W_P(x)                             # (B, N, A, D)
        x = x.reshape(B * N, x.size(2), x.size(3))  # (B * N, A, D)
        return x

class DynamicPatcherMoS(nn.Module):
    """
    Mixture-of-Strides: 여러 (patch_len, stride) 브랜치를 병렬 구성 후 게이팅 가중합.
    input: x (B, L, N)
    output: (B * N, A_max, D)
    attributes: patch_num = A_max, d_model = D
    """
    def __init__(self,
                 base_configs: PatchMixerConfigMonthly,
                 branches=((4, 2), (8, 4), (12, 6)),
                 d_model: int = 256,
                 gate_hidden: int = 64,
                 ):
        super().__init__()
        self.branches = nn.ModuleList()
        self.patch_nums = []
        self.d_model = int(d_model)

        # gate: (B, K) - 시간 축 평균 후 N개 채널을 요약
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),    # (B, N, L) -> (B, N, 1)
            nn.Flatten(1, 2),
            nn.Linear(base_configs.enc_in, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, len(branches)),
        )

        for P, S in branches:
            cfg = copy.deepcopy(base_configs)
            cfg.patch_len, cfg.stride = int(P), int(S)
            self.branches.append(SimpleUnfoldProjector(cfg, d_model = self.d_model))
            self.patch_nums.append(self.branches[-1].patch_num)

        self.patch_num = max(self.patch_nums) # A_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N)
        B, L, N = x.shape

        # Gate: (B, K)
        # gate 입력은 (B, N, L) 필요
        xn = x.permute(0, 2, 1)     # (B, N, L)
        g = torch.softmax(self.gate_net(xn), dim = -1) # (B, K)

        # 각 분기 추출 후 (A_max)로 padding
        Z_list = []
        for k, patcher in enumerate(self.branches):
            z = patcher(x) # (B*N, A_K, D)
            if z.size(1) < self.patch_num:
                pad = (0, 0, 0, self.patch_num - z.size(1)) # (last two dims)
                z = F.pad(z, pad)
            Z_list.append(z.unsqueeze(1))   # (B * N, 1, A_max, D)

        Z = torch.cat(Z_list, dim = 1) # (B * N, K, A_max, D)

        # gate를 (B * N, K, 1, 1)로 broadcast
        g_bn = g.repeat_interleave(N, dim = 0).unsqueeze(-1).unsqueeze(-1) # (B * N, K, 1, 1)
        Z = (g_bn * Z).sum(dim = 1) # (B*N, A_max, D)
        return Z

class DynamicOffsetPatcher(nn.Module):
    """
    Learnable Offsets: anchor A개에 대해 offset을 예측, 1D grid sample로 (A, P) 윈도우 샘플링
    input: x (B, L, N)
    output: (B * N, A, D), reg_loss(정규화 항) 별도 반환 필요 시 외부에서 합산
    attributes: patch_num = A, d_model = D
    """
    def __init__(self,
                 cfg: PatchMixerConfigMonthly,
                 d_model: int = 256,
                 stride_base: int = 4,
                 patch_len: int = 8,
                 max_off: float = 4.0,
                 lambda_off: float = 1e-3,
                 lambda_tv: float = 1e-3,
                 ):
        super().__init__()
        self.P = int(patch_len)
        self.S = int(stride_base)
        base = int((cfg.lookback - self.P) / self.S + 1)
        self.patch_num = base + 1
        self.max_off = float(max_off)
        self.lambda_off = float(lambda_off)
        self.lambda_tv = float(lambda_tv)

        self.off_head = nn.Sequential(
            nn.Conv1d(cfg.enc_in, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, 32, 1)
            nn.Flatten(),             # (B, 32)
            nn.Linear(32, self.patch_num), # (B, A)
        )

        self.W_P = nn.Linear(self.P, self.d_model)
        self.register_buffer('patch_idx', torch.arange(self.P).float())
        self._last_reg: torch.Tensor | None = None

    def regularizer(self, delta: torch.Tensor) -> torch.Tensor:
        l1 = delta.abs().mean()
        tv = (delta[:, 1:] - delta[:, :-1]).abs().mean() if delta.size(1) > 1 else torch.tensor(0.0, device = delta.device)
        return self.lambda_off * l1 + self.lambda_tv * tv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N)
        B, L, N = x.shape
        anchors = torch.arange(self.patch_num, device = x.device, dtype = torch.float32) * self.S # (A,)
        delta = self.max_off * torch.tanh(self.off_head(x.permute(0, 2, 1)))  # (B, A)

        xs = anchors.view(1, -1, 1) + delta.unsqueeze(-1) + (self.patch_idx.view(1, 1, -1) - (self.P - 1) / 2)
        xs = xs.clamp(0, L - 1)                  # (B, A, P)
        x_norm = 2.0 * (xs / max(1, L-1)) - 1.0       # [-1, 1]
        y_norm = torch.zeros_like(x_norm)             # (B, A, P)
        grid = torch.stack([x_norm, y_norm], dim = -1) # (B, A, P, 2)

        z = x.permute(0, 2, 1).reshape(B * N, 1, 1, L) # (B*N, 1, 1, L)
        samp = F.grid_sample(
            z, grid.repeat_interleave(N, dim = 0),
            mode = 'bilinear', padding_mode = 'border', align_corners = True
        ) # (B * N, 1, A, P)

        samp = samp.squeeze(1)    # (B * N, A, P)
        out = self.W_P(samp)      # (B * N, A, D)

        self._last_reg = self.regularizer(delta)
        return out

    def last_reg_loss(self) -> torch.Tensor:
        return self._last_reg if self._last_reg is not None else torch.tensor(0.0)