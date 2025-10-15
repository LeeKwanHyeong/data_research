import copy
import torch

from torch.amp import autocast, GradScaler

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.losses import LossComputer
from modeling_module.training.optim import build_optimizer_and_scheduler


class CommonTrainer:
    def __init__(self,
                 cfg,
                 adapter,
                 *,
                 metrics_fn=None,
                 logger=print,
                 future_exo_cb=None
                 ):  # exo 콜백
        self.cfg = cfg
        self.adapter: DefaultAdapter = adapter
        self.logger = logger
        self.loss_comp = LossComputer(cfg)
        self.metrics_fn = metrics_fn
        self.future_exo_cb = future_exo_cb

        self.amp_enabled = (self.cfg.amp_device == "cuda" and torch.cuda.is_available())

    def _to_tensor(self, x, device):
        if x is None:
            raise RuntimeError(
                "[Loss None] loss is None. Check LossComputer.compute() and model output.\n"
                f"- amp_device={self.cfg.amp_device}, device={device}"
            )
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _make_future_exo(self, x, y, *, device):
        """
        future_exo_cb가 있으면 (B, H, exo_dim) 텐서를 만들어 반환.
           없으면 None.
        H는 타깃 y의 길이로 맞춥니다.
        """
        if self.future_exo_cb is None:
            return None
        B = x.size(0)
        H = y.size(1)  # 타깃 horizon
        t0 = 0  # 필요 시 데이터셋에서 시작 인덱스를 꺼내도록 확장 가능
        exo = self.future_exo_cb(t0, H, device=device)     # (H, exo_dim)
        exo = exo.unsqueeze(0).expand(B, -1, -1).to(device)  # (B, H, exo_dim)
        return exo

    def _run_epoch(self, model, loader, *, train: bool):
        device = self.cfg.device
        total = 0.0
        model.train() if train else model.eval()

        with torch.set_grad_enabled(train):
            for batch in loader:
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x, y = x.to(device), y.to(device)

                if train:
                    self.opt.zero_grad(set_to_none=True)

                # exo 생성 (필요 시)
                future_exo = self._make_future_exo(x, y, device=device)

                with autocast(self.cfg.amp_device, enabled=self.amp_enabled):
                    # kwarg로 안전 전달
                    pred = self.adapter.forward(
                        model,
                        x,
                        future_exo=future_exo,
                        mode=("train" if train else "eval"),
                    )
                    loss = self.loss_comp.compute(pred, y, is_val=not train)
                    r = self.adapter.reg_loss(model)
                    if r is not None:
                        loss = loss + r

                if train:
                    loss_t = self._to_tensor(loss, device)
                    if torch.isnan(loss_t):
                        self.logger("[Warn] NaN loss. step skipped.")
                        continue
                    self.scaler.scale(loss_t).backward()
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
        self.scaler = GradScaler(self.cfg.amp_device)

        best_loss = float('inf')
        best_state = copy.deepcopy(model.state_dict())
        counter = 0

        if self.adapter.uses_tta():
            self.adapter.tta_reset(model)

        for epoch in range(self.cfg.epochs):
            train_loss = self._run_epoch(model, train_loader, train=True)

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

                    # exo 생성 (검증 시에도 동일하게)
                    future_exo = self._make_future_exo(x_val, y_val, device=device)

                    if tta_steps > 0 and self.adapter.uses_tta():
                        # TTA는 기본적으로 x만 사용 (필요하면 어댑터/TTM 쪽 확장)
                        loss = self.adapter.tta_adapt(model, x_val, y_val, steps=tta_steps)
                        if loss is None:
                            with autocast(self.cfg.amp_device, enabled=self.amp_enabled):
                                pred = self.adapter.forward(
                                    model, x_val,
                                    future_exo=future_exo, mode="eval"
                                )
                                loss = self.loss_comp.compute(pred, y_val, is_val=True)
                                loss = float(loss.detach())
                        val_total += loss
                    else:
                        with autocast(self.cfg.amp_device, enabled=self.amp_enabled):
                            pred = self.adapter.forward(
                                model, x_val,
                                future_exo=future_exo, mode="eval"
                            )
                            vloss = self.loss_comp.compute(pred, y_val, is_val=True)
                            val_total += float(vloss.detach())

                    if self.metrics_fn:
                        _ = self.metrics_fn(pred, y_val)

            val_loss = val_total / max(1, len(val_loader))
            self.sched.step()

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