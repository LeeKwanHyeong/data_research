from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Callable

import torch
from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import TrainingConfig, StageConfig
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.losses import make_pspa_fn
from modeling_module.utils.exogenous_utils import calendar_sin_cos


def _dump_cfg(cfg: TrainingConfig):
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_patchtst] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def train_patchtst(
    model,
    train_loader,
    val_loader,
    *,
    stages: list[StageConfig] | None = None,
    train_cfg: TrainingConfig,
    future_exo_cb: Optional[Callable] = None,
    exo_is_normalized: bool = True,   # (향후 사용 예정: 외생변수 정규화 플래그)
):
    """
    PatchTST도 PatchMixer/Titan과 동일한 2-Stage 학습 스케줄을 지원합니다.
    - stages: [StageConfig(...), StageConfig(...)]
    - train_cfg: 공통 TrainingConfig (loss_mode='auto' 권장)
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."
    stages = stages or [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    # AMP 설정 (Titan/PM과 일치)
    amp_device = getattr(train_cfg, "amp_device", "cuda")
    amp_enabled = (amp_device == "cuda" and torch.cuda.is_available())
    amp_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(str(getattr(train_cfg, "amp_dtype", "bf16")).lower(), torch.bfloat16)

    autocast_input = {
        "device_type": amp_device,
        "enabled": amp_enabled,
        "dtype": amp_dtype,
    }

    best = {"model": model, "cfg": train_cfg}

    for i, stg in enumerate(stages, start=1):
        # Stage별로 cfg 복제/주입
        cfg_i = train_cfg.copy_with(
            epochs=stg.epochs,
            lr=stg.lr if stg.lr is not None else train_cfg.lr,
            use_horizon_decay=bool(stg.use_horizon_decay),
            tau_h=stg.tau_h if stg.tau_h is not None else train_cfg.tau_h,
            spike_loss=train_cfg.spike_loss.with_enabled(stg.spike_enabled),
        )

        print(
            f"[train_patchtst] ===== Stage {i}/{len(stages)} ===== "
            f"- spike: {'ON' if stg.spike_enabled else 'OFF'} "
            f"- epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}"
        )
        _dump_cfg(cfg_i)

        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=DefaultAdapter(),          # PatchTST는 기본 어댑터로 충분
            future_exo_cb=future_exo_cb or calendar_sin_cos,  # 주/월 sin-cos 자동 주입
            logger=print,
            autocast_input=autocast_input,
            # extra_loss_fn=make_pspa_fn()
            extra_loss_fn=None
        )
        model = trainer.fit(model, train_loader, val_loader, tta_steps=0)
        best = {"model": model, "cfg": cfg_i}

    return best
