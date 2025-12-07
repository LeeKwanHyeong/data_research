import torch
from torch import nn

from modeling_module.models.PatchTST.common.patching import compute_patch_num
from modeling_module.models.PatchTST.common.pos_encoding import positional_encoding, PositionalEncoding
from modeling_module.models.common_layers.RevIN import RevIN


class PatchBackboneBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.c_in = cfg.c_in
        self.d_model = cfg.d_model
        self.patch_len = cfg.patch_len
        self.stride = cfg.stride
        # patch 수 계산
        from modeling_module.models.PatchTST.common.patching import compute_patch_num
        self.patch_num = compute_patch_num(cfg.lookback, cfg.patch_len, cfg.stride, cfg.padding_patch)

        # patch projection layer
        self.patch_proj = nn.Conv1d(
            in_channels=self.c_in,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            stride=self.stride,
        )

        # positional encoding
        from modeling_module.models.PatchTST.common.pos_encoding import positional_encoding
        self.pos_enc = positional_encoding(
                pe=getattr(cfg, "pe", "sincos"),          # sincos 또는 zeros
                learn_pe=getattr(cfg, "learn_pe", True),  # 학습 여부
                q_len=self.patch_num,                     # patch 개수
                d_model=self.d_model,                     # feature 차원
            )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력: x [B, C, L]
        출력: z [B, L_patch, d_model]
        """
        # 1) Convolution (Patching)
        # padding_patch 처리는 do_patch 유틸을 쓰거나 여기서 처리해야 함.
        # PatchTST는 보통 unfold 대신 Conv1d(stride)를 써서 패칭과 임베딩을 동시에 함.

        # 만약 입력 길이 L이 stride로 나누어 떨어지지 않거나 패딩이 필요하다면
        # Conv1d가 알아서 처리하지 않으므로 사전에 패딩이 필요할 수 있음.
        # 여기서는 단순하게 Conv1d 결과 사용

        z = self.patch_proj(x)  # [B, d_model, L_patch]
        z = z.transpose(1, 2)  # [B, L_patch, d_model]

        # 2) Positional Encoding 더하기
        # 현재 배치의 패치 개수(L_patch)에 맞춰 PE 슬라이싱
        B, L_patch, D = z.shape

        # self.pos_enc는 [Max_Len, D] 형태라고 가정
        if self.pos_enc.size(0) < L_patch:
            # 미리 생성된 PE보다 긴 시퀀스가 들어오면 새로 생성 (동적 대응)
            new_pe = PositionalEncoding(L_patch, D).to(z.device)
            # 학습 파라미터가 아니라면 바로 사용 가능, 학습 파라미터라면 문제될 수 있음
            # (PatchTST는 보통 SinCos 고정 PE를 쓰므로 새로 생성해도 무방)
            z = z + new_pe.unsqueeze(0)
        else:
            # 길이만큼 잘라서 사용
            z = z + self.pos_enc[:L_patch, :].unsqueeze(0)

        return z