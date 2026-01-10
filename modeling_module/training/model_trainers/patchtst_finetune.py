# modeling_module/training/model_trainers/patchtst_finetune.py
from __future__ import annotations

import os
from typing import Optional, Callable, Dict, Any

import torch

from modeling_module.training.config import TrainingConfig, StageConfig
from modeling_module.training.model_trainers.patchtst_train import train_patchtst


def _load_pretrain_state(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        # 사용자가 encoder만 저장했다면(예: export_encoder_state)
        return ckpt["encoder"]
    if isinstance(ckpt, dict):
        # state_dict 자체를 저장했을 가능성
        # (다만 키가 모델 파라미터 형태인지 확인 필요)
        return ckpt
    raise ValueError(f"Unsupported ckpt format: {type(ckpt)}")


def _freeze_encoder_blocks(model) -> None:
    """
    pretrain으로 학습된 encoder를 잠깐 고정하고 head만 적응시키고 싶을 때 사용.
    아래 prefix는 본 대화에서 만든 self_supervised backbone 기준입니다.
    프로젝트 supervised PatchTST의 모듈 명칭에 맞게 필요시 조정하세요.
    """
    freeze_prefixes = (
        "backbone.patch_embed.",
        "backbone.encoder.",
        "backbone.norm_out.",
    )
    for n, p in model.named_parameters():
        if any(n.startswith(px) for px in freeze_prefixes):
            p.requires_grad = False


def _unfreeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = True


def train_patchtst_finetune(
    model,
    train_loader,
    val_loader,
    *,
    train_cfg: Optional[TrainingConfig] = None,
    stages: list[StageConfig] | None = None,
    # pretrain loading
    pretrain_ckpt_path: Optional[str] = None,
    load_strict: bool = False,
    # optional freeze
    freeze_encoder_before_ft: bool = False,
    unfreeze_after_stage0: bool = True,
    # exo passthrough (기존 patchtst_train과 동일)
    future_exo_cb: Optional[Callable] = None,
    exo_is_normalized: bool = True,
) -> Dict[str, Any]:
    """
    Supervised fine-tune runner:
      1) (optional) load self-supervised pretrained weights into supervised model
      2) (optional) freeze encoder for stage0
      3) call existing train_patchtst()

    Notes:
      - 실제 supervised 학습 로직은 patchtst_train.train_patchtst()를 그대로 사용합니다.
      - pretrain ckpt 키가 supervised 모델과 완전히 같지 않을 수 있으므로 strict=False를 기본 권장합니다.
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."

    # 1) load pretrain ckpt (if provided)
    if pretrain_ckpt_path is not None:
        if not os.path.exists(pretrain_ckpt_path):
            raise FileNotFoundError(pretrain_ckpt_path)

        state = _load_pretrain_state(pretrain_ckpt_path)
        missing, unexpected = model.load_state_dict(state, strict=load_strict)

        print("[Finetune] loaded pretrain ckpt:", pretrain_ckpt_path)
        print(f"[Finetune] load_strict={load_strict}")
        print(f"[Finetune] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if len(missing) > 0:
            print("  - missing (first 20):", missing[:20])
        if len(unexpected) > 0:
            print("  - unexpected (first 20):", unexpected[:20])

    # 2) optional freeze encoder
    if freeze_encoder_before_ft:
        _freeze_encoder_blocks(model)
        print("[Finetune] encoder blocks frozen (stage0).")

    # 3) supervised training via existing pipeline
    out = train_patchtst(
        model,
        train_loader,
        val_loader,
        stages=stages,
        train_cfg=train_cfg,
        future_exo_cb=future_exo_cb,
        exo_is_normalized=exo_is_normalized,
    )

    # 4) optionally unfreeze after stage0 (다음 stage를 돌리기 전에 쓰고 싶으면,
    #    stages를 2개로 쪼개서 stage0 후 호출하는 형태로 사용하는 것을 권장)
    if freeze_encoder_before_ft and unfreeze_after_stage0:
        _unfreeze_all(model)
        print("[Finetune] encoder blocks unfrozen (post stage0).")

    return out