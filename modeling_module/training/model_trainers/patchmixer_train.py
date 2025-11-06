# patchmixer_train.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Callable

import torch
from modeling_module.training.adapters import PatchMixerAdapter, DefaultAdapter
from modeling_module.training.config import TrainingConfig
from modeling_module.training.engine import CommonTrainer

def _dump_cfg(cfg):
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_patchmixer] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))

def train_patchmixer(
    model,
    train_loader,
    val_loader,
    *,
    train_cfg: Optional[TrainingConfig] = None,
    # 외생변수
    future_exo_cb: Optional[Callable[[int,int], "torch.Tensor"]] = None,
    exo_is_normalized: bool = True,
):
    """
    - Titan과 동일한 사용성: train_cfg 있으면 그 값을 우선 사용
    - AMP 설정/어댑터/외생변수 콜백 전달을 공통 트레이너에 위임
    """

    _dump_cfg(train_cfg)

    # AMP 설정(bf16 기본)
    amp_device = getattr(train_cfg, "amp_device", "cuda")
    amp_enabled = (amp_device == "cuda" and torch.cuda.is_available())
    amp_dtype_str = getattr(train_cfg, "amp_dtype", "bf16")
    if isinstance(amp_dtype_str, torch.dtype):
        amp_dtype = amp_dtype_str
    else:
        s = str(amp_dtype_str).lower()
        amp_dtype = (
            torch.bfloat16 if s in ("bf16", "bfloat16")
            else torch.float16 if s in ("fp16","float16","half")
            else torch.float32 if s in ("fp32","float32")
            else torch.bfloat16
        )

    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    # PatchMixerAdapter 사용 (필요 시 내부에서 exo_is_normalized 힌트 활용 가능)
    adapter = PatchMixerAdapter() if PatchMixerAdapter else DefaultAdapter()

    trainer = CommonTrainer(
        cfg=train_cfg,
        adapter=adapter,
        logger=print,
        metrics_fn=None,
        future_exo_cb=future_exo_cb,  # ← (B,H,D_exo) 생성은 트레이너가 처리
        autocast_input=autocast_input,
    )
    best_model = trainer.fit(model, train_loader, val_loader, tta_steps=0)

    print(
        f"[EXO-train] model.exo_dim={getattr(model, 'exo_dim', 0)}  "
        f"future_exo_cb? {future_exo_cb is not None}  "
        f"exo_is_normalized={exo_is_normalized}"
    )
    return {
        "model": best_model,
        "cfg": train_cfg,
    }
