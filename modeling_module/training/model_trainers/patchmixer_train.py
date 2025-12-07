# patchmixer_train.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn  # NEW: exo_head 재구성을 위해

from torch.utils.data import DataLoader, WeightedRandomSampler

from modeling_module.training.adapters import PatchMixerAdapter, DefaultAdapter
from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.losses import make_pspa_fn


def _dump_cfg(cfg):
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_patchmixer] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))

def _infer_exo_dim_from_cb(future_exo_cb, horizon: int, device: str = "cpu") -> int:
    if future_exo_cb is None:
        return 0
    fe = future_exo_cb(0, horizon, device=device)  # (H,E) 가정
    if isinstance(fe, torch.Tensor):
        return int(fe.size(-1))
    try:
        # numpy/리스트 등
        return int(fe.shape[-1])
    except Exception:
        return 0

def _ensure_model_exo_head(model, exo_dim: int):
    """
    model.exo_dim 과 exo_dim(E)가 다르면 exo_head를 재구성.
    """
    # 모델이 exo_head를 제공하지 않는다면 스킵
    if not hasattr(model, "exo_dim"):
        return model

    current = int(getattr(model, "exo_dim", 0))
    if exo_dim <= 0:
        # 외생 안 쓰는 경우: exo_head 제거
        if getattr(model, "exo_head", None) is not None:
            model.exo_head = None
        model.exo_dim = 0
        return model

    # 이미 일치하면 패스
    if current == exo_dim and getattr(model, "exo_head", None) is not None:
        return model

    # (재)구성
    model.exo_head = nn.Sequential(
        nn.Linear(exo_dim, 64),
        nn.GELU(),
        nn.Linear(64, 1)
    )
    model.exo_dim = int(exo_dim)
    print(f"[train_patchmixer] exo_head rebuilt with exo_dim={exo_dim}")
    return model


def _maybe_make_spike_loader(train_loader: DataLoader, enable: bool) -> DataLoader:
    """
    2단계에서만 스파이크 샘플을 더 자주 보게 하는 간단 오버샘플러.
    Dataset에 `sample_is_spike` (bool array-like)가 있으면 가중 샘플러 적용.
    """
    if (not enable) or (not hasattr(train_loader.dataset, "sample_is_spike")):
        return train_loader

    import numpy as np
    m = np.asarray(train_loader.dataset.sample_is_spike, dtype=bool)
    w = np.where(m, 3.0, 1.0).astype("float32")  # spike:others = 3:1
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


def train_patchmixer(
    model,
    train_loader,
    val_loader,
    *,
    stages: list[StageConfig] | None = None,
    train_cfg: Optional[TrainingConfig] = None,
    # 외생변수
    future_exo_cb: Optional[Callable[[int, int], "torch.Tensor"]] = None,
    exo_is_normalized: bool = True,
):
    """
    Titan과 동일한 사용성:
      - stages 가 없으면 기존처럼 train_cfg 한 번만 학습
      - stages 가 있으면 각 StageConfig로 train_cfg를 덮어써서 연속 학습
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."

    # (A) 여기서 exo_dim(E) 자동 추론 후, 모델의 exo_head를 미리 세팅
    horizon = getattr(model, "horizon", None) or getattr(train_cfg, "horizon", None)
    if horizon is None:
        raise ValueError("horizon을 model 또는 train_cfg에서 찾을 수 없습니다.")
    E = _infer_exo_dim_from_cb(future_exo_cb, horizon, device="cpu")
    model = _ensure_model_exo_head(model, E)
    print(
        f"[EXO-setup] inferred E={E}, model.exo_dim={getattr(model, 'exo_dim', None)}, has_head={model.exo_head is not None}")

    # AMP 설정(bf16 기본)
    amp_device = getattr(train_cfg, "amp_device", "cuda")
    amp_enabled = (amp_device == "cuda" and torch.cuda.is_available())
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
    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    # Adapter (PatchMixer 전용이 있으면 사용)
    adapter = PatchMixerAdapter() if PatchMixerAdapter else DefaultAdapter()

    # 스테이지 목록 구성 (없으면 단일 스테이지)
    if not stages or len(stages) == 0:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    best = None
    for i, stg in enumerate(stages, 1):
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_patchmixer] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        _dump_cfg(cfg_i)

        # (선택) 2단계에서만 오버샘플링 적용
        tl_i = _maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            logger=print,
            metrics_fn=None,
            future_exo_cb=future_exo_cb,   # (B,H,E) 외생변수 콜백
            autocast_input=autocast_input,
            # extra_loss_fn=make_pspa_fn()
            extra_loss_fn=None,
        )
        model = trainer.fit(model, tl_i, val_loader, tta_steps=0)
        best = {"model": model, "cfg": cfg_i}

    print(
        f"[EXO-train] model.exo_dim={getattr(model, 'exo_dim', 0)}  "
        f"future_exo_cb? {future_exo_cb is not None}  "
        f"exo_is_normalized={exo_is_normalized}"
    )
    return best
