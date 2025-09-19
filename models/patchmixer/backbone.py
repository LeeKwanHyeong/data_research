# -------------------------
# PatchMixer Backbone
# -------------------------
import copy

import torch
import torch.nn as nn

from models.layers.RevIN import RevIN


# -------------------------
# PatchMixer Layer
# -------------------------
class PatchMixerLayer(nn.Module):
    """
    Depthwise(패치 축 로컬) + 1x1(Pointwise, Channel Mixing)로 구성된 기본 Mixer 블록.
    입력/출력 텐서 shape: (N, C = patch_num, L = d_model)
    """
    def __init__(self,
                 dim: int,
                 a: int,
                 kernel_size: int = 0,
                 dropout: float = 0.0
                 ):
        super().__init__()
        self.dim = dim # in/out channels (= patch_num)
        self.a = a     # out channels for 1x1 (normally match with patch_num)
        self.kernel_size = kernel_size

        # Depthwise Conv over length(L=d_model)
        self.token_mixer = nn.Sequential(
            nn.Conv1d(
                in_channels = dim,
                out_channels = dim,
                kernel_size = kernel_size,
                groups = dim,
                padding = 'same'
            ),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )

        # Pointwise 1x1 Conv for channel mixing
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size = 1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C=patch_num, L = d_model)
        """
        # Residual over depthwise branch (shape 동일)
        x = x + self.token_mixer(x)
        x = self.channel_mixer(x)
        x = self.dropout(x)
        return x


class PatchMixerBackbone(nn.Module):
    """
    input: (B, L = lookback, N = n_vars)
    output: (B, a * d_model) # Global patch representation(mean variable concatenate)
    """
    def __init__(self,
                 configs,
                 revin: bool = True,
                 affine: bool = True,
                 subtract_last: bool = False,
                 ):
        super().__init__()
        self.configs = configs

        # basic hyperparameter
        self.n_vals = configs.enc_in
        self.lookback = configs.lookback
        self.forecasting = configs.horizon
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.kernel_size = configs.mixer_kernel_size

        # patch num calculation (+1은 패딩으로 1 patch 더 확보)
        base = int((self.lookback - self.patch_size) / self.stride + 1)
        self.patch_num = base + 1
        self.a = self.patch_num
        self.d_model = configs.d_model
        self.dropout_rate = configs.head_dropout
        self.depth = configs.e_layers

        # output dimension (representation size of backbone output)
        self.patch_repr_dim = self.a * self.d_model

        # unfold after padding (끝단 복제를 통해 마지막 패치 확보)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        # PatchMixer blocks
        self.PatchMixer_blocks = nn.ModuleList([
            PatchMixerLayer(dim = self.patch_num, a = self.a, kernel_size = self.kernel_size, dropout = self.dropout_rate)
            for _ in range(self.depth)
        ])

        # patch length -> model dimension linear projection (각 패치를 d_model로 투영)
        self.W_P = nn.Linear(self.patch_size, self.d_model)

        # Normalization(RevIN)
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.n_vals, affine = affine, subtract_last = subtract_last)

        self.flatten = nn.Flatten(start_dim =- 2) # (C, L) -> (C*L)

    @torch.no_grad()
    def _assert_input_shape(self, x: torch.Tensor) -> None:
        # expectation: (B, L, N)
        if x.dim() != 3:
            raise ValueError(f"Expected input 3D tensor (B, L, N). Got shape = {tuple(x.shape)}")
        if x.size(1) != self.lookback:
            # Rather than strictly checking, just warning
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flow of PatchMixer
        -> x: (B, L, N)
        -> RevIn(norm)
        -> (B, N, L)
        -> unfold(patch)
        -> (B*N, patch_num, d_model)
        -> PatchMixer blocks
        -> flatten
        -> (B, a*d_model)
        """
        bs, seq_len, n_vars = x.shape
        self._assert_input_shape(x)

        # RevIN Normalization (B, L, N)
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # (B, N, L)
        x = x.permute(0, 2, 1)

        # patch unfold after padding: (B, N, patch_num, patch_size)
        x_lookback = self.padding_patch_layer(x) # (B, N, L + stride)
        x = x_lookback.unfold(dimension = -1, size = self.patch_size, step = self.stride)

        # linear projection: 마지막 축 patch_size -> d_model
        x = self.W_P(x) # (B, N, patch_num, d_model)

        # 변수별 독립 처리 위해 (B*N, patch_num, d_model)로 reshaping
        x = x.reshape(bs * n_vars, x.size(2), x.size(3))

        for block in self.PatchMixer_blocks:
            x = block(x) # (B*N, patch_num, d_model)

        # Global representation: (B*N, patch_num * d_model) -> (B, N, -1) -> mean variable
        x = self.flatten(x)        # (B*N, patch_num * d_model)
        x = x.view(bs, n_vars, -1) # (B, N, patch_num * d_model)
        x = x.mean(dim = 1)        # (B, a * d_model) 변수 축 평균 집약

        return x # (B, patch_repr_dim)

    def inverse_norm(self, y: torch.Tensor) -> torch.Tensor:
        """필요 시 예측치(또는 재구성치)에 대한 RevIN 역변환 지원용 헬퍼"""
        if not self.revin:
            return y
        return self.revin_layer(y, 'denorm')

class MultiScalePatchMixerBackbone(nn.Module):
    """
    서로 다른 (patch_len, stride, kernel) 분기를 병렬 구성.
    - RevIN은 이 래퍼에서 1회 적용 -> 분기 내부는 revin = False
    - 각 분기 출력 (a_i * d_model) -> per-branch Linear로 per_branch_dim 정렬 -> 융합
    """
    def __init__(self,
                 base_configs,
                 patch_cfgs: ((4, 2, 5), (8, 4, 7), (12, 6, 9)),  # (patch_len, stride, kernel)
                 per_branch_dim: int = 128,
                 fused_dim: int = 256,
                 fusion: str = 'concat',  # ['concat', 'gated']
                 affine: bool = True,
                 subtract_last: bool = False,
                 ):
        super().__init__()
        self.fusion = fusion
        self.branches = nn.ModuleList()
        self.projs = nn.ModuleList()

        # 공유 RevIN: 입력을 한 번만 정규화
        self.revin = RevIN(base_configs.enc_in, affine = affine, subtract_last = subtract_last)

        for (pl, st, ks) in patch_cfgs:
            cfg = copy.deepcopy(base_configs)
            cfg.patch_len = pl
            cfg.stride = st
            cfg.mixer_kernel_size = ks
            branch = PatchMixerBackbone(cfg, revin = False) # 내부 RevIN 비활성화
            self.branches.append(branch)
            self.projs.append(nn.Linear(branch.patch_repr_dim, per_branch_dim))

        if fusion == 'concat':
            self.fuse = nn.Linear(per_branch_dim * len(self.branches), fused_dim)
        elif fusion == 'gated':
            self.fuse = nn.Linear(per_branch_dim, fused_dim)
            self.gate = nn.Linear(per_branch_dim, 1)
        else:
            raise ValueError("fusion must be 'concat' or 'gated'")

        self.out_dim = fused_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, N) -> (B, fused_dim)
        """
        x = self.revin(x, 'norm')
        reps = []
        gates = []
        for branch, proj in zip(self.branches, self.projs):
            b = branch(x) # (B, a_i * d_model)
            b = proj(b)   # (B, per_branch_dim)
            reps.append(b)
            if self.fusion == 'gated':
                gates.append(self.gate(b)) # (B, 1)

        if self.fusion == 'concat':
            z = torch.cat(reps, dim = 1)    # (B, per_branch_dim * n_branch)
            z = self.fuse(z)                # (B, fused_dim)
        else:
            G = torch.softmax(torch.cat(gates, dim = 1), dim = 1) # (B, n_branch)
            S = torch.stack(reps, dim = 1)                        # (B, n_branch, per_branch_dim)
            z = (G.unsqueeze(-1) * S).sum(dim = 1)                # (B, per_branch_dim)
            z = self.fuse(z)                                      # (B, fused_dim)
        return z


class PatchMixerBackboneWithPatcher(nn.Module):
    """
    외부 패처(Module)가 만들어 준 (B*N, A, D)를 받아 PatchMixer blocks를 통과시키는 Backbone.
    input: (B, L, N)
    output: (B, A*D) *변수 축 평균 집약
    """
    def __init__(self,
                 configs,
                 patcher: nn.Module,  # DynamicPatcherMoS or DynamicOffsetPatcher 등
                 e_layers: int | None = None,  # block 수 오버라이드 가능
                 dropout_rate: float | None = None,
                 ):
        super().__init__()
        self.cfg = configs
        self.n_vals = configs.enc_in
        self.horizon = configs.horizon

        # Patcher Meta
        self.patcher = patcher
        self.a: int = int(getattr(patcher, 'patch_num'))
        self.d_model: int = int(getattr(patcher, 'd_model'))
        self.patch_repr_dim: int = self.a * self.d_model

        self.depth = int(e_layers if e_layers is not None else configs.e_layers)
        self.dropout_rate = float(dropout_rate if dropout_rate is not None else configs.head_dropout)

        # Mixer blocks (Conv1d는 (N, C=patch_num, L=d_model) 포맷)
        self.blocks = nn.ModuleList([
            PatchMixerLayer(dim=self.a, a=self.a, kernel_size=configs.mixer_kernel_size, dropout=self.dropout_rate)
            for _ in range(self.depth)
        ])

        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N)
        B, L, N = x.shape

        z = self.patcher(x)  # (B*N, A, D)
        for blk in self.blocks:
            z = blk(z)  # (B*N, A, D)

        z = self.flatten(z)  # (B*N, A*D)
        z = z.view(B, N, -1).mean(1)  # (B, A*D)
        return z