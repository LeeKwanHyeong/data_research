# modeling_module/training/model_trainers/patchtst_pretrain.py
from __future__ import annotations

import os
import json
from dataclasses import asdict, is_dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage


def _dump_cfg(cfg, save_dir: Optional[str], name: str) -> None:
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    payload = asdict(cfg) if is_dataclass(cfg) else dict(cfg.__dict__)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _getattr(cfg, key: str, default):
    return getattr(cfg, key, default)


def _extract_x(batch):
    """
    프로젝트 데이터셋이 (x, y, ..., fe_cont, ...) 형태일 수 있으므로,
    self-supervised에서는 x만 안전하게 뽑습니다.
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _to_device(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


@torch.no_grad()
def _eval_pretrain(model, loader: DataLoader, device: torch.device, *, mask_ratio: float, loss_type: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = _to_device(_extract_x(batch), device)
        out = model(x, mask_ratio=mask_ratio, return_loss=True, loss_type=loss_type)
        loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
        total += float(loss.detach().cpu().item())
        n += 1
    return total / max(n, 1)


def train_patchtst_pretrain(
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    stages: list[StageConfig] | None = None,
    train_cfg: Optional[TrainingConfig] = None,
    # self-supervised knobs
    mask_ratio: float = 0.3,
    loss_type: str = "mse",
    # io
    save_dir: Optional[str] = None,
    ckpt_name: str = "patchtst_pretrain_best.pt",
):
    """
    Self-supervised pretrain runner (Masked Patch Reconstruction)

    Requirements:
      - model(x, mask_ratio=..., return_loss=True, loss_type=...) -> dict with ["loss"]
        (본 대화에서 만든 PatchTSTPretrainModel 시그니처 기준)

    Notes:
      - TrainingConfig/StageConfig는 patchtst_train.py 흐름을 그대로 맞추기 위해 유지합니다.
      - 다만 실제 학습 루프는 CommonTrainer에 의존하지 않고 torch loop로 구현합니다.
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."
    stages = stages or [StageConfig()]  # 프로젝트 정책에 맞게 기본 stage 1개

    device = torch.device(_getattr(train_cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(_getattr(train_cfg, "use_amp", _getattr(train_cfg, "amp", True)))
    grad_clip = float(_getattr(train_cfg, "grad_clip", 0.0))
    log_every = int(_getattr(train_cfg, "log_every", 100))

    model = model.to(device)

    best_val = float("inf")
    best_state = None

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _dump_cfg(train_cfg, save_dir, "pretrain_cfg.json")

    for si, stg in enumerate(stages):
        cfg_i = apply_stage(train_cfg, stg)
        epochs = int(_getattr(cfg_i, "epochs", _getattr(cfg_i, "max_epochs", 10)))
        lr = float(_getattr(cfg_i, "lr", 1e-3))
        weight_decay = float(_getattr(cfg_i, "weight_decay", 0.0))

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

        print(f"[Pretrain] stage={si} epochs={epochs} lr={lr} wd={weight_decay} mask_ratio={mask_ratio} loss={loss_type}")

        for ep in range(1, epochs + 1):
            model.train()
            running = 0.0
            step = 0

            for batch in train_loader:
                step += 1
                x = _to_device(_extract_x(batch), device)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    out = model(x, mask_ratio=mask_ratio, return_loss=True, loss_type=loss_type)
                    loss = out["loss"] if isinstance(out, dict) and "loss" in out else out

                scaler.scale(loss).backward()

                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()

                running += float(loss.detach().cpu().item())

                if (step % log_every) == 0:
                    print(f"[Pretrain][stage={si} ep={ep}/{epochs} step={step}] loss={running/step:.6f}")

            train_loss = running / max(step, 1)

            if val_loader is not None:
                val_loss = _eval_pretrain(model, val_loader, device, mask_ratio=mask_ratio, loss_type=loss_type)
                print(f"[Pretrain][stage={si} ep={ep}/{epochs}] train={train_loss:.6f} val={val_loss:.6f}")

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if save_dir is not None:
                        ckpt_path = os.path.join(save_dir, ckpt_name)
                        torch.save(
                            {"state_dict": best_state, "best_val": best_val, "cfg": asdict(cfg_i) if is_dataclass(cfg_i) else None},
                            ckpt_path,
                        )
            else:
                print(f"[Pretrain][stage={si} ep={ep}/{epochs}] train={train_loss:.6f}")

    # 마지막에 best_state를 model에 다시 반영
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    print(f"[Pretrain] done | best_val={best_val:.6f}" if val_loader is not None else "[Pretrain] done")
    return {"model": model, "best_val": best_val if val_loader is not None else None}