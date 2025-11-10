# -------------------------
# PatchMixer Backbone
# -------------------------
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
class FeatureAlign(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.fc: nn.Linear | None = None
        self.in_dim: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        if (self.fc is None) or (self.in_dim != d):
            # 입력 차원이 달라지면 in_dim을 갱신하며 선형층을 재생성
            self.fc = nn.Linear(d, self.out_dim, bias=True).to(x.device)
            self.in_dim = d
        return self.fc(x)
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
            nn.Conv1d(d_model, d_model, kernel_size=self.ks, stride=1, padding=0,
                      dilation=self.dilation, groups=d_model),  # depthwise
            nn.GELU(),
            nn.GroupNorm(num_groups=min(32, d_model), num_channels=d_model),  # ← 변경
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=min(32, d_model), num_channels=d_model),  # ← 변경
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
    input:  (B, L=lookback, N=n_vars)
    output: (B, A * D)  (변수축 평균 집약)
    - out_dim: A * d_model  (※ 모델 초기화 시점의 lookback/patch_len/stride로 결정)
    """
    def __init__(self, configs, revin: bool = True, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.configs = configs

        # hyper
        self.n_vals: int = int(configs.enc_in)
        self.lookback: int = int(configs.lookback)
        self.patch_size: int = int(configs.patch_len)
        self.stride: int = int(configs.stride)
        self.d_model: int = int(configs.d_model)
        self.depth: int = int(configs.e_layers)
        self.dropout_rate: float = float(getattr(configs, "head_dropout", 0.0))

        # 패치 수 계산: unfold(L + stride, size=P, step=stride)와 동일 결과
        base = (self.lookback - self.patch_size) / float(self.stride) + 1.0
        base = int(base)  # floor
        self.patch_num: int = base + 1  # 패딩으로 +1
        if self.patch_num < 1:
            self.patch_num = 1

        # ★ 공식 출력 차원 (init 기준 고정)
        self.patch_repr_dim: int = self.patch_num * self.d_model
        self.out_dim: int = self.patch_repr_dim  # <- 모델이 참조할 단일 진실값

        # unfold용 패딩(끝단 복제): +stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        # 블록
        self.blocks = nn.ModuleList([
            PatchMixerLayer(d_model=self.d_model, kernel_size=int(configs.mixer_kernel_size), dropout=self.dropout_rate)
            for _ in range(self.depth)
        ])

        # 패치 → d_model 투영
        self.W_P = nn.Linear(self.patch_size, self.d_model)

        self.flatten = nn.Flatten(start_dim=-2)  # (C, L) -> (C*L)

    @torch.no_grad()
    def _assert_3d(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"Expected input 3D tensor (B, L, N). Got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flow
          (B, L, N) -> (B, N, L)
          pad(+stride) & unfold(size=patch_size, step=stride) -> (B,N,A,P)
          W_P: (B,N,A,D)
          reshape (B*N, A, D) -> permute (B*N, D, A)
          PatchMixer blocks
          flatten -> (B*N, D*A) -> (B, N, D*A) -> mean(N) -> (B, D*A)
        """
        self._assert_3d(x)
        B, L, N = x.shape
        if L != self.lookback:
            if L > self.lookback:
                # 뒤에서 자르기 (최근 구간 사용)
                x = x[:, -self.lookback:, :]
            else:
                # 왼쪽을 복제 패딩해서 길이를 맞춤
                pad = self.lookback - L
                # time 차원이 두번째(=dim=1)이니, transpose로 (B,N,L)로 바꿔 1D pad 후 복원
                x = x.transpose(1, 2)  # (B, N, L)
                x = F.pad(x, (pad, 0), mode="replicate")  # (B, N, lookback)
                x = x.transpose(1, 2)  # (B, lookback, N)
        # (B, N, L)
        x = x.permute(0, 2, 1)

        # 패딩 & 언폴드
        x = self.padding_patch_layer(x)                     # (B, N, L + stride)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # (B, N, A_eff, P)

        # 실제 패치 수(런타임)
        A_eff = x.size(2)

        # size P → D
        x = self.W_P(x)                                     # (B, N, A_eff, D)

        # (B*N, D, A_eff)
        BNA = x.reshape(B * N, A_eff, self.d_model)
        BDA = BNA.permute(0, 2, 1)

        for blk in self.blocks:
            BDA = blk(BDA)

        # (B*N, D*A_eff) -> (B, N, D*A_eff)
        z = self.flatten(BDA).view(B, N, -1)

        # 변수축 평균
        z = z.mean(dim=1)                                   # (B, D*A_eff)

        # 개발 중 차원 변동 감지용 경고(학습/추론 중 lookback 불일치 등)
        if z.size(-1) != self.out_dim:
            # 경고만. 상위 모듈에서 z_proj로 보정.
            if self.training:
                print(f"[PatchMixerBackbone][warn] forward out_dim={z.size(-1)} "
                      f"!= declared out_dim={self.out_dim} "
                      f"(A_eff={A_eff}, A_cfg={self.patch_num})")

        return z  # (B, D*A_eff)



class MultiScalePatchMixerBackbone(nn.Module):
    """
    서로 다른 (patch_len, stride, kernel) 분기를 병렬 구성.
    각 분기: (B, A_i*D) → per_branch_dim 정렬 → 융합
    out_dim = fused_dim
    """
    def __init__(
        self,
        base_configs,
        patch_cfgs: tuple = ((4, 2, 5), (8, 4, 7), (12, 6, 9)),
        per_branch_dim: int = 128,
        fused_dim: int = 256,
        fusion: str = "concat",  # ['concat', 'gated']
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.fusion = fusion
        self.branches = nn.ModuleList()
        self.projs = nn.ModuleList()

        for (pl, st, ks) in patch_cfgs:
            cfg = copy.deepcopy(base_configs)
            cfg.patch_len = int(pl)
            cfg.stride = int(st)
            cfg.mixer_kernel_size = int(ks)

            branch = PatchMixerBackbone(cfg, revin=False)
            self.branches.append(branch)
            self.projs.append(nn.Linear(branch.out_dim, per_branch_dim))  # Lazy 대신 명시형

        if fusion == "concat":
            self.fuse = nn.Linear(per_branch_dim * len(self.branches), fused_dim)
        elif fusion == "gated":
            self.fuse = nn.Linear(per_branch_dim, fused_dim)
            self.gate = nn.Linear(per_branch_dim, 1)
        else:
            raise ValueError("fusion must be 'concat' or 'gated'")

        self.out_dim = fused_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps, gates = [], []
        for branch, proj in zip(self.branches, self.projs):
            b = branch(x)         # (B, A_i*D_i)
            b = proj(b)           # (B, per_branch_dim)
            reps.append(b)
            if self.fusion == "gated":
                gates.append(self.gate(b))  # (B,1)

        if self.fusion == "concat":
            z = torch.cat(reps, dim=1)      # (B, per_branch_dim * n_branch)
            z = self.fuse(z)                # (B, fused_dim)
        else:
            G = torch.softmax(torch.cat(gates, dim=1), dim=1)  # (B, n_branch)
            S = torch.stack(reps, dim=1)                       # (B, n_branch, per_branch_dim)
            z = (G.unsqueeze(-1) * S).sum(dim=1)               # (B, per_branch_dim)
            z = self.fuse(z)                                   # (B, fused_dim)
        return z




class PatchMixerBackboneWithPatcher(nn.Module):
    """
    외부 patcher가 만들어 준 (B*N, A, D)를 받아 PatchMixer blocks를 통과.
    output: (B, A*D) (변수축 평균)
    """
    def __init__(self, configs, patcher: nn.Module, e_layers: int | None = None, dropout_rate: float | None = None):
        super().__init__()
        self.cfg = configs
        self.patcher = patcher

        self.a: int = int(getattr(patcher, "patch_num"))
        self.d_model: int = int(getattr(patcher, "d_model"))
        self.patch_repr_dim: int = self.a * self.d_model
        self.out_dim: int = self.patch_repr_dim

        self.depth = int(e_layers if e_layers is not None else configs.e_layers)
        self.dropout_rate = float(dropout_rate if dropout_rate is not None else configs.head_dropout)

        self.blocks = nn.ModuleList([
            PatchMixerLayer(d_model=self.d_model, kernel_size=int(configs.mixer_kernel_size), dropout=self.dropout_rate)
            for _ in range(self.depth)
        ])
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N = x.shape
        z = self.patcher(x)  # (B*N, A, D)
        for blk in self.blocks:
            z = blk(z)       # (B*N, D, A) <- 주의: patcher 포맷에 맞추어야 하면 permute 필요

        # 위에서 PatchMixerLayer는 (B*, D, A)를 가정하므로 patcher가 (B*, A, D)를 낸다면 여기서 permute 하세요.
        # z = z.permute(0, 2, 1)  # 필요 시

        z = self.flatten(z)             # (B*N, A*D)
        z = z.view(B, N, -1).mean(1)    # (B, A*D)
        return z