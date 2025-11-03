# engine_refactored.py
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import torch
from torch.amp import autocast, GradScaler

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.losses import LossComputer
from modeling_module.training.optim import build_optimizer_and_scheduler


'''
손실 계산의 단일 경로화
LossComputer.compute(pred, y, is_val=...)로만 계산. 내부에서 point/quantile/auto 분기를 처리하고, 
간헐수요 가중·허버·핀볼·뉴스벤더 q* 등을 알아서 적용. (기존 LossComputer를 그대로 사용)

혼합정밀 / 그래드 누적 / 클리핑
self.grad_accum_steps(기본 1)로 누적을 제어하고, AMP + 클리핑을 항상 같은 순서로 처리해 학습안정성을 보장.

Exogenous
학습/검증 모두 future_exo_cb(t0, H)로 (B,H,exo_dim) 만들고, adapter.forward(..., future_exo=...)로 전달. 
모델이 외생변수를 지원하지 않으면 어댑터가 무시.

콜백 확장성
TrainerCallbacks 로 on_epoch_start/end, on_batch_end를 주입해 WandB/파일로깅/중간시각화 같은 부가기능을 깔끔하게 분리.

ReduceLROnPlateau 호환
scheduler.step(val_loss) → Plateau류, 그렇지 않으면 step() 호출로 일반 스케줄러 모두 호환.

'''


# ---------------------------
# Optional Callback Interface
# ---------------------------
@dataclass
class TrainerCallbacks:
    on_epoch_start: Optional[Callable[[int, Dict[str, Any]], None]] = None
    on_epoch_end: Optional[Callable[[int, Dict[str, Any]], None]] = None
    on_batch_end: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    # 예: wandb 로깅, 중간 시각화, 샘플별 디버깅 등


class CommonTrainer:
    """
    깔끔한 공통 학습 엔진
     - adapter.forward() 1곳에서만 모델 호출
     - LossComputer 1곳에서만 손실 계산
     - AMP/Grad Accum/Clip 표준화
     - future_exo_cb 로 (B,H,exo_dim) 생성
     - 콜백으로 유연한 로깅/메트릭 확장
    """
    def __init__(
        self,
        cfg,
        adapter: DefaultAdapter,
        *,
        logger: Callable[[str], None] = print,
        metrics_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
        future_exo_cb: Optional[Callable[[int, int, torch.device], torch.Tensor]] = None,
        callbacks: Optional[TrainerCallbacks] = None,
    ):
        self.cfg = cfg
        self.adapter = adapter
        self.logger = logger
        self.loss_comp = LossComputer(cfg)
        self.metrics_fn = metrics_fn
        self.future_exo_cb = future_exo_cb
        self.cb = callbacks or TrainerCallbacks()
        self.amp_enabled = (getattr(cfg, "amp_device", "cpu") == "cuda" and torch.cuda.is_available())

        # 선택 옵션(없으면 기본값)
        self.grad_accum = int(getattr(cfg, "grad_accum_steps", 1))
        self.max_grad_norm = float(getattr(cfg, "max_grad_norm", 1_000.0))
        self.patience = int(getattr(cfg, "patience", 10))
        self.epochs = int(getattr(cfg, "epochs", 20))
        self.device = getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # -------- util --------
    @staticmethod
    def _as_tensor(x, device):
        if x is None:
            raise RuntimeError("[Loss None] LossComputer.compute()가 None을 반환했습니다.")
        return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32, device=device)

    def _make_future_exo(self, x: torch.Tensor, y: torch.Tensor, device) -> Optional[torch.Tensor]:
        """
        (B, H, exo_dim) 텐서. H는 타깃 길이(y.size(1)) 기준.
        없으면 None.
        """
        if self.future_exo_cb is None:
            return None
        B = x.size(0)
        H = y.size(1)
        exo = self.future_exo_cb(0, H, device)  # 필요시 시작 인덱스는 데이터셋에서 넘기도록 확장
        return exo.unsqueeze(0).expand(B, -1, -1).to(device)

    # -------- one pass --------
    def _run_loader(self, model, loader, *, train: bool) -> float:
        device = self.device
        model.train(mode=train)
        scaler = self.scaler if train else None

        total_loss = 0.0
        step_in_accum = 0

        for bi, batch in enumerate(loader):
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device)
            y = y.to(device)

            future_exo = self._make_future_exo(x, y, device=device)

            # forward + loss (AMP)
            with torch.set_grad_enabled(train), autocast(getattr(self.cfg, "amp_device", "cpu"), enabled=self.amp_enabled):
                pred = self.adapter.forward(
                    model, x, future_exo=future_exo, mode=("train" if train else "eval")
                )
                loss = self.loss_comp.compute(pred, y, is_val=not train)

            loss_t = self._as_tensor(loss, device)

            if train:
                # grad accumulation
                loss_t = loss_t / self.grad_accum
                scaler.scale(loss_t).backward()
                step_in_accum += 1

                if step_in_accum >= self.grad_accum:
                    scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    scaler.step(self.opt)
                    scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    step_in_accum = 0

            total_loss += float(loss_t.detach()) * (self.grad_accum if train else 1)

            # per-batch callback/metrics
            if self.cb.on_batch_end or self.metrics_fn:
                info = {"loss": float(loss_t.detach())}
                if self.metrics_fn:
                    with torch.no_grad():
                        try:
                            m = self.metrics_fn(pred, y) or {}
                            info.update({f"m/{k}": float(v) for k, v in m.items()})
                        except Exception:
                            pass
                if self.cb.on_batch_end:
                    self.cb.on_batch_end(bi, 1 if not train else self.grad_accum, info)

        return total_loss / max(1, len(loader))

    # -------- fit --------
    def fit(self, model, train_loader, val_loader, *, tta_steps: int = 0):
        device = self.device
        model.to(device)

        self.opt, self.sched = build_optimizer_and_scheduler(model, self.cfg)
        self.scaler = GradScaler(getattr(self.cfg, "amp_device", "cpu"))

        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0

        # (선택) TTA 준비
        if self.adapter.uses_tta():
            self.adapter.tta_reset(model)

        for epoch in range(1, self.epochs + 1):
            if self.cb.on_epoch_start:
                self.cb.on_epoch_start(epoch, {})

            train_loss = self._run_loader(model, train_loader, train=True)

            # ---- Validation ----
            model.eval()
            with torch.no_grad():
                val_loss = self._run_loader(model, val_loader, train=False)

            # 스케줄러 스텝(Plateau 사용 시에는 val_loss 기준으로 step)
            try:
                self.sched.step(val_loss)  # ReduceLROnPlateau 호환
            except TypeError:
                self.sched.step()          # Cosine, StepLR 등

            # early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.logger(f"[EarlyStop] epoch={epoch}, best_val={best_loss:.6f}")
                    break

            # 로그
            cur_lr = self.sched.get_last_lr()[0] if hasattr(self.sched, "get_last_lr") else self.opt.param_groups[0]["lr"]
            self.logger(f"Epoch {epoch}/{self.epochs} | LR {cur_lr:.6f} | Train {train_loss:.6f} | Val {val_loss:.6f}")

            if self.cb.on_epoch_end:
                self.cb.on_epoch_end(epoch, {"train_loss": train_loss, "val_loss": val_loss, "lr": cur_lr})

        model.load_state_dict(best_state)
        return model
