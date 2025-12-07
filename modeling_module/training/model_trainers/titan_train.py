from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Callable

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from modeling_module.training.adapters import TitanAdapter, DefaultAdapter
from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.losses import make_pspa_fn
from modeling_module.utils.exogenous_utils import calendar_sin_cos


def _pick_future_exo_cb(model, user_cb: Optional[Callable]) -> Optional[Callable]:
    """우선순위: 사용자 콜백 > 모델이 캘린더 사용 → calendar_sin_cos > 없음"""
    if user_cb is not None:
        return user_cb

    use_calendar = False
    exo_dim = int(getattr(model, "exo_dim", 0))

    model_cfg = getattr(model, "config", None)
    if model_cfg is not None:
        use_calendar = bool(getattr(model_cfg, "use_calendar_exo", False))
        exo_dim = int(getattr(model_cfg, "exo_dim", exo_dim))

    if not use_calendar:
        use_calendar = (exo_dim > 0)

    return calendar_sin_cos if use_calendar else None


def _dump_cfg(cfg):
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_titan] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _maybe_make_spike_loader(train_loader: DataLoader, enable: bool) -> DataLoader:
    """
    2단계에서만 스파이크 샘플을 더 자주 보게 하는 간단 오버샘플러.
    Dataset에 `sample_is_spike` (bool array-like)가 있으면 가중 샘플러 적용.
    """
    if (not enable) or (not hasattr(train_loader.dataset, "sample_is_spike")):
        return train_loader

    import numpy as np
    m = np.asarray(train_loader.dataset.sample_is_spike, dtype=bool)
    w = np.where(m, 3.0, 1.0).astype("float32")  # 스파이크:기본 = 3:1
    sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

    return DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        num_workers=train_loader.num_workers,
        pin_memory=getattr(train_loader, "pin_memory", True),
        drop_last=getattr(train_loader, "drop_last", False),
        collate_fn=getattr(train_loader, "collate_fn", None),
    )


def train_titan(
    model,
    train_loader,
    val_loader,
    *,
    stages: list[StageConfig] | None = None,
    train_cfg: Optional[TrainingConfig] = None,
    future_exo_cb=None,
):
    """
    2-stage 커리큘럼 지원:
      - stages 가 없으면 기존처럼 train_cfg 한 번만 학습
      - stages 가 있으면 각 StageConfig로 train_cfg를 덮어써서 연속 학습
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."

    # 외생변수 콜백 자동 선택(사용자 지정이 우선)
    future_exo_cb = _pick_future_exo_cb(model, future_exo_cb)

    # AMP 설정
    amp_device = getattr(train_cfg, "amp_device", "cuda")
    amp_dtype_str = getattr(train_cfg, "amp_dtype", "bf16")
    if isinstance(amp_dtype_str, torch.dtype):
        amp_dtype = amp_dtype_str
    else:
        s = str(amp_dtype_str).lower()
        if s in ("bf16", "bfloat16"):
            amp_dtype = torch.bfloat16
        elif s in ("fp16", "float16", "half"):
            amp_dtype = torch.float16
        elif s in ("fp32", "float32"):
            amp_dtype = torch.float32
        else:
            amp_dtype = torch.bfloat16
    amp_enabled = (amp_device == "cuda" and torch.cuda.is_available())
    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    adapter = DefaultAdapter()  # 필요 시 TitanAdapter로 교체

    # 스테이지 목록 구성 (없으면 단일 스테이지)
    if not stages or len(stages) == 0:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    best = None
    for i, stg in enumerate(stages, 1):
        # 스테이지별 cfg 적용
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_titan] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        _dump_cfg(cfg_i)

        # (선택) 2단계에서만 오버샘플링 적용
        tl_i = _maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        # 스테이지 트레이너 생성 및 실행
        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            future_exo_cb=future_exo_cb,
            logger=print,
            autocast_input=autocast_input,
            # extra_loss_fn=make_pspa_fn()
            extra_loss_fn = None
        )
        model = trainer.fit(model, tl_i, val_loader, tta_steps=2)
        best = {"model": model, "cfg": cfg_i}

    return best
