from typing import Optional

import torch.nn as nn

def compute_patch_num(
        lookback: int,
        patch_len: int,
        stride: int,
        padding_patch: Optional[str] = None # 'end' or None
) -> int:
    """
    patch 개수 계산:
        - 기본: n = floor((lookback - patch_len) / stride) + 1
        - padding_patch == 'end' 인 경우, 마지막 패치를 하나 더 보장
    """
    if lookback < patch_len:
        raise ValueError(f"lookback({lookback}) must be >= patch_len({patch_len})")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, gor {stride}")
    if padding_patch not in (None, 'end'):
        raise ValueError("padding_patch must be one of {None, 'end'}")

    n = (lookback - patch_len) // stride + 1
    if padding_patch == 'end':
        n += 1
    return n

def do_patch(
        x,
        patch_len: int,
        stride: int,
        padding_patch: Optional[str] = None
):
    """
    x: [B, n_Vars, lookback] -> [B, n_vars, patch_len, patch_num]
    """
    if padding_patch == 'end':
        pad = nn.ReplicationPad1d((0, stride))
        x = pad(x)
    x = x.unfold(dimension=-1, size = patch_len, step = stride) # [B, n_vars, patch_num, patch_len]
    return x.permute(0, 1, 3, 2)                                # [B, n_vars, patch_len, patch_num]

def unpatch(x_patched):
    # x_patched: [B, n_vars, patch_len, patch_num] -> 복원 유틸
    raise NotImplementedError
