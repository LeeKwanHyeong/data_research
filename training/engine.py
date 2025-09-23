import copy, torch
import torch.optim as optim
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from training.losses import LossComputer
from training.optim import build_optimizer_and_scheduler


class CommonTrainer:
    def __init__(self, cfg, adapter, *, metrics_fn = None, logger = print):
        self.cfg = cfg
        self.adapter = adapter
        self.logger = logger
        self.loss_comp = LossComputer(cfg)
        self.metrics_fn = metrics_fn # (pred, y) -> dict or None

    def _run_epoch(self, model, loader, *, train: bool):
        device = self.cfg.device
        total = 0.0
        if train: model.train()
        else: model.eval()
        with torch.set_grad_enabled(train):
            for batch in loader:
                if len(batch) == 3:
                    x, y, _ = batch

                else:
                    x, y = batch
                x, y = x.to(device), y.to(device)

                if train:
                    self.opt.zero_grad(set_to_none=True)

                with autocast(self.cfg.amp_device):
                    pred = self.adapter.forward(model, x)
                    loss = self.loss_comp.compute(pred, y, is_val = not train)
                    r = self.adapter.reg_loss(model)
                    if r is not None:
                        loss = loss + r

                if train:
                    if torch.isnan(loss):
                        self.logger("[Warn] Nan loss. step skipped.")
                        continue
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                total += float(loss.detach())
        return total / max(1, len(loader))

    def fit(self, model, train_loader, val_loader, *, tta_steps: int = 0):
        device = self.cfg.device
        model.to(device)
        self.opt, self.sched = build_optimizer_and_scheduler(model, self.cfg)
        self.scaler = GradScaler()

        best_loss = float('inf')
        best_state = copy.deepcopy(model.state_dict())
        counter = 0

        # TTA Initialize
        if self.adapter.uses_tta():
            self.adapter.tta_rest(model)

        for epoch in range(self.cfg.epochs):
            # Warming Up: 간헐 가중 alpha_zero 서서히 키우고 싶다면 여기서 cfg 동적 조절
            train_loss = self._run_epoch(model, train_loader, train = True)

            # ---- Validation ----
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        x_val, y_val, _ = batch
                    else:
                        x_val, y_val = batch
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    if tta_steps > 0 and self.adapter.uses_tta():
                        # Titan: Memory-as-Context
                        loss = self.adapter.tta_adapt(model, x_val, y_val, steps = tta_steps)
                        if loss is None:
                            # fallback: 일반 검증
                            with autocast(self.cfg.amp_device):
                                pred = self.adapter.forward(model, x_val)
                                loss = self.loss_comp.compute(pred, y_val, is_val = True)
                                loss = float(loss.detach())
                        val_total += loss
                    else:
                        with autocast(self.cfg.amp_device):
                            pred = self.adapter.forward(model, x_val)
                            vloss = self.loss_comp.compute(pred, y_val, is_val = True)
                            val_total += float(vloss.detach())

                    if self.metrics_fn:
                        m = self.metrics_fn(pred, y_val)
            val_loss = val_total / max(1, len(val_loader))

            # Scheduler
            self.sched.step()

            # Early Stopping
            if val_loss < best_loss:
                best_loss, counter = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= self.cfg.patience:
                    self.logger(f"Early stopping at epoch {epoch+1}")
                    break

            cur_lr = self.sched.get_last_lr()[0]
            self.logger(f"Epoch {epoch+1}/{self.cfg.epochs} | LR {cur_lr:.6f} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        model.load_state_dict(best_state)
        return model
