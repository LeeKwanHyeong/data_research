# -------------------------
# PatchMixer Backbone
# -------------------------
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchMixerLayer(nn.Module):
    """
    입력: (B*, D=d_model, A=patch_num)
    - depthwise conv는 채널(D) 기준, 길이는 A 방향
    - 출력 shape 동일: (B*, D, A)
    """
    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        self.d_model = d_model
        self.ks = int(kernel_size)
        self.dilation = int(dilation)

        # Conv는 padding=0로 두고, forward에서 동적으로 pad
        self.token_mixer = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model,
                kernel_size=self.ks, stride=1, padding=0,  # ★ 0
                dilation=self.dilation, groups=d_model  # depthwise
            ),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _same_pad_1d(self, L: int) -> tuple[int, int]:
        # SAME padding 총량 = dil*(ks-1)
        total = self.dilation * (self.ks - 1)
        left = total // 2
        right = total - left
        return left, right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*, D, A)
        res = x
        L = x.size(-1)
        pl, pr = self._same_pad_1d(L)
        if pl or pr:
            x = F.pad(x, (pl, pr))   # (left, right)
        x = self.token_mixer(x)      # (B*, D, A) 길이 보존
        x = self.channel_mixer(x)
        x = self.dropout(x)
        return x + res                # 길이 동일 → 문제 없음


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
        self.n_vals: int = configs.enc_in
        self.lookback: int = configs.lookback
        self.forecasting: int = configs.horizon
        self.patch_size: int = configs.patch_len
        self.stride: int = configs.stride
        self.kernel_size: int = configs.mixer_kernel_size

        # patch num calculation (+1은 패딩으로 1 patch 더 확보)
        base = int((self.lookback - self.patch_size) / self.stride + 1)
        self.patch_num: int = base + 1
        self.a: int = self.patch_num
        self.d_model: int = configs.d_model
        self.dropout_rate: float = configs.head_dropout
        self.depth: int = configs.e_layers
        # output dimension (representation size of backbone output)
        self.patch_repr_dim = self.a * self.d_model

        # unfold after padding (끝단 복제를 통해1 마지막 패치 확보)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        # PatchMixer blocks
        self.PatchMixer_blocks = nn.ModuleList([
            PatchMixerLayer(d_model=self.d_model, kernel_size=configs.mixer_kernel_size, dropout=configs.head_dropout)
            for _ in range(configs.e_layers)
        ])

        # patch length -> model dimension linear projection (각 패치를 d_model로 투영)
        self.W_P = nn.Linear(configs.patch_len, self.d_model)

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
        # if self.revin:
        #     x = self.revin_layer(x, 'norm')

        # (B, N, L)
        x = x.permute(0, 2, 1)

        # patch unfold after padding: (B, N, patch_num, patch_size)
        x_lookback = self.padding_patch_layer(x) # (B, N, L + stride)
        x = x_lookback.unfold(dimension = -1, size = self.patch_size, step = self.stride)

        # linear projection: 마지막 축 patch_size -> d_model
        x = self.W_P(x) # (B, N, patch_num, d_model)
        # print("after W_P:", x.shape)
        actual_patch_num = x.shape[2]  # 실제 patch 개수를 동적으로 가져오기
        self.patch_num = actual_patch_num  # 내부 변수 업데이트
        self.a = self.patch_num
        self.patch_repr_dim = self.a * self.d_model
        # print("after W_P:", x.shape)
        # # 변수별 독립 처리 위해 (B*N, patch_num, d_model)로 reshaping
        # x = x.reshape(bs * n_vars, x.size(2), x.size(3))
        #
        # for block in self.PatchMixer_blocks:
        #     x = block(x) # (B*N, patch_num, d_model)
        #
        # # Global representation: (B*N, patch_num * d_model) -> (B, N, -1) -> mean variable
        # x = self.flatten(x)        # (B*N, patch_num * d_model)
        # x = x.view(bs, n_vars, -1) # (B, N, patch_num * d_model)
        # assert x.shape[-1] == self.patch_num * self.d_model, f"Unexpected feature dim: {x.shape[-1]}"
        # x = x.mean(dim = 1)        # (B, a * d_model) 변수 축 평균 집약

        # (B*N, D, A)로 변환해 레이어 통과
        BNA = x.reshape(bs * n_vars, self.patch_num, self.d_model)  # (B*N, A, D)
        BDA = BNA.permute(0, 2, 1)                                   # (B*N, D, A)
        # print("before blocks:", BDA.shape)
        for blk in self.PatchMixer_blocks:
            BDA = blk(BDA)                                           # (B*N, D, A)

        # (B*N, D*A) → (B, N, D*A)
        x = self.flatten(BDA)                                        # (B*N, D*A)
        x = x.view(bs, n_vars, -1)                                   # (B, N, D*A)
        # print("after flatten+view:", x.shape, " expected last=", self.patch_num * self.d_model)
        assert x.shape[-1] == self.patch_num * self.d_model, f"Unexpected feature dim: {x.shape[-1]} != {self.patch_num*self.d_model}"

        # 변수 축 pooling
        x = x.mean(dim=1)

        return x # (B, patch_repr_dim)


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

        for (pl, st, ks) in patch_cfgs:
            cfg = copy.deepcopy(base_configs)
            cfg.patch_len = pl
            cfg.stride = st
            cfg.mixer_kernel_size = ks
            branch = PatchMixerBackbone(cfg, revin = False) # 내부 RevIN 비활성화
            self.branches.append(branch)
            self.projs.append(nn.LazyLinear(per_branch_dim))

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
            PatchMixerLayer(patch_num=self.a, d_model=self.d_model, kernel_size=configs.mixer_kernel_size, dropout=self.dropout_rate)
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