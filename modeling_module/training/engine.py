import copy
import torch
from torch.amp import autocast, GradScaler

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.losses import LossComputer


class CommonTrainer:
    """
    - LossComputer를 감싼 트레이너
    - spike_loss.mode = "mix" 인 경우: LossComputer.compute_with_spike_mix() 경유
    - point_loss = "huber_asym" 인 경우: LossComputer.compute() 내부에서 처리
    """
    def __init__(
        self,
        cfg,
        adapter: DefaultAdapter,
        *,
        metrics_fn=None,
        logger=print,
        future_exo_cb=None,
        autocast_input = None,
    ):
        self.cfg = cfg
        self.adapter: DefaultAdapter = adapter
        self.logger = logger
        self.loss_comp = LossComputer(cfg)
        self.metrics_fn = metrics_fn
        self.future_exo_cb = future_exo_cb
        self.amp_enabled = (self.cfg.amp_device == "cuda" and torch.cuda.is_available())
        self.autocast_input = autocast_input

        if autocast_input is not None:
            self.amp_device = autocast_input['device_type']
            self.enabled = autocast_input['enabled']
            self.dtype = autocast_input['dtype']


        # self.amp_enabled = False

    # ----------------- 내부 유틸 -----------------
    @staticmethod
    def _to_tensor(x, device):
        if x is None:
            raise RuntimeError(
                "[Loss None] loss is None. Check LossComputer and model output."
            )
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _make_future_exo(self, x, y, *, device):
        """
        future_exo_cb가 무엇을 반환하든 최종적으로 (B, H, E)로 정규화한다.
          허용 입력: (H,E), (1,H,E), (B,H,E), (1,1,H,E) 등
        """
        if self.future_exo_cb is None:
            return None

        B = x.size(0)
        H = y.size(1)  # 타깃 horizon 기준
        t0 = 0

        exo = self.future_exo_cb(t0, H, device=device)
        # ---- shape 정규화 ----
        if not torch.is_tensor(exo):
            raise TypeError(f"future_exo_cb must return torch.Tensor, got {type(exo)}")

        # squeeze 가능한 앞쪽 1 차원은 모두 제거 (예: [1,1,H,E] -> [H,E])
        while exo.dim() >= 3 and exo.size(0) == 1:
            exo = exo.squeeze(0)

        if exo.dim() == 2:
            # (H,E) -> (1,H,E)
            exo = exo.unsqueeze(0)
        elif exo.dim() == 3:
            # (B' or 1, H, E) 허용. B'==1 이면 브로드캐스트
            pass
        elif exo.dim() == 4:
            # (1,1,H,E) 같은 케이스를 강제 정규화
            if exo.size(0) == 1 and exo.size(1) == 1:
                exo = exo.squeeze(0).squeeze(0)  # -> (H,E)
                exo = exo.unsqueeze(0)  # -> (1,H,E)
            else:
                raise RuntimeError(f"future_exo_cb returned 4D exo with unexpected shape {tuple(exo.shape)}")
        else:
            raise RuntimeError(
                f"future_exo_cb returned tensor with unsupported dim={exo.dim()}, shape={tuple(exo.shape)}")

        # 이제 exo는 (1 or B', H, E). B'가 1이면 배치로 expand
        if exo.size(0) == 1 and B > 1:
            exo = exo.expand(B, -1, -1)
        elif exo.size(0) != B and exo.size(0) != 1:
            # 배치 크기가 딱 맞지 않으면 에러
            raise RuntimeError(f"[EXO] batch size mismatch: exo.shape[0]={exo.size(0)} vs B={B}")

        exo = exo.to(device)

        # 1회만 로깅
        if not hasattr(self, "_logged_exo_shape"):
            print(f"[EXO-batch] exo normalized to shape={tuple(exo.shape)} (expect [B,H,E])")
            self._logged_exo_shape = True

        return exo

    def _compute_loss(self, pred, y, *, is_val: bool):
        return self.loss_comp.compute(pred, y, is_val=is_val)

    def _nan_stat(self, name, t):
        if not torch.is_tensor(t):
            return
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()

        # 버전 호환: torch.nanmax 사용하지 않고 finite 마스크로 최대 절댓값을 계산
        finite_mask = torch.isfinite(t)
        if finite_mask.any():
            try:
                mx = t[finite_mask].abs().max().item()
            except Exception:
                # 일부 dtype(정수, bool) 대비 방어
                mx = t[finite_mask].to(torch.float32).abs().max().item()
        else:
            mx = float('inf')  # 전부 NaN/Inf인 경우

        if has_nan or has_inf:
            print(f"[NaN-{name}] has_nan={has_nan} has_inf={has_inf} max|x|={mx}")

    # ----------------- 에폭 루프 -----------------
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

                self._nan_stat("x(in)", x)
                self._nan_stat("y(in)", y)


                if train:
                    self.opt.zero_grad(set_to_none=True)

                future_exo = self._make_future_exo(x, y, device=device)
                if future_exo is not None:
                    # 외생변수 가드: NaN→0, Inf→유한값
                    future_exo = torch.nan_to_num(future_exo, nan=0.0, posinf=1e6, neginf=-1e6)
                    self._nan_stat("future_exo", future_exo)

                with autocast(
                        device_type = self.cfg.amp_device,
                        enabled=self.amp_enabled,
                        dtype = self.dtype if self.dtype is not None else 'fp32'
                ):
                    pred = self.adapter.forward(
                        model,
                        x,
                        future_exo=future_exo,
                        mode=("train" if train else "eval"),
                    )
                    self._nan_stat("pred", pred)
                    loss = self._compute_loss(pred, y, is_val=(not train))
                    self._nan_stat("loss_raw", loss)
                    reg = self.adapter.reg_loss(model)
                    if reg is not None:
                        self._nan_stat("reg", reg)
                        loss = loss + reg

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

    # ----------------- 학습 진입 -----------------
    def fit(self, model, train_loader, val_loader, *, tta_steps: int = 0):
        device = self.cfg.device
        model.to(device)
        from modeling_module.training.optim import build_optimizer_and_scheduler
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

                    future_exo = self._make_future_exo(x_val, y_val, device=device)

                    if tta_steps > 0 and self.adapter.uses_tta():
                        loss = self.adapter.tta_adapt(model, x_val, y_val, steps=tta_steps)
                        if loss is None:
                            with autocast(
                                    device_type=self.cfg.amp_device,
                                    enabled=self.amp_enabled,
                                    dtype=self.dtype if self.dtype is not None else 'fp32'

                            ):
                                pred = self.adapter.forward(
                                    model, x_val, future_exo=future_exo, mode="eval"
                                )
                                loss = self._compute_loss(pred, y_val, is_val=True)
                                loss = float(loss.detach())
                        val_total += loss
                    else:
                        with autocast(
                                device_type=self.cfg.amp_device,
                                enabled=self.amp_enabled,
                                dtype=self.dtype if self.dtype is not None else 'fp32'
                        ):
                            pred = self.adapter.forward(
                                model, x_val, future_exo=future_exo, mode="eval"
                            )
                            vloss = self._compute_loss(pred, y_val, is_val=True)
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
