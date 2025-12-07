import copy
import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Tuple

import torch
from torch.amp import autocast, GradScaler

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.losses import LossComputer


class CommonTrainer:
    """
    - LossComputer를 감싼 트레이너
    - 배치 형식:
        (x, y, part_ids, future_exo_cont, past_exo_cont, past_exo_cat)  # 권장(6)
        (x, y, part_ids, future_exo_cont, past_exo_cont)                # (5)
        (x, y, part_ids)                                                # (3)
        (x, y)                                                          # (2)
    - future_exo: 데이터셋 제공분이 우선, 없으면 future_exo_cb로 생성
    - past_exo_cont / past_exo_cat / part_ids는 adapter.forward로 전달
    """
    def __init__(
        self,
        cfg,
        adapter: DefaultAdapter,
        *,
        metrics_fn=None,
        logger=print,
        future_exo_cb=None,
        autocast_input=None,
        extra_loss_fn=None,
    ):
        self.cfg = cfg
        self.adapter: DefaultAdapter = adapter
        self.logger = logger
        self.loss_comp = LossComputer(cfg)
        self.metrics_fn = metrics_fn
        self.future_exo_cb = future_exo_cb
        self.amp_enabled = (self.cfg.amp_device == "cuda" and torch.cuda.is_available())
        self.autocast_input = autocast_input or {}
        self.extra_loss_fn = extra_loss_fn

        # autocast 설정
        self.amp_device = self.autocast_input.get("device_type", self.cfg.amp_device)
        self.enabled = self.autocast_input.get("enabled", self.amp_enabled)
        self.dtype = self.autocast_input.get("dtype", None)

        def _dump(obj, title):
            data = asdict(obj) if is_dataclass(obj) else obj.__dict__
            self.logger(f"[CommonTrainer] {title}")
            self.logger(json.dumps(data, indent=2, ensure_ascii=False, default=str))

        _dump(self.cfg, "TrainingConfig (final)")
        if hasattr(self.adapter, "cfg"):
            _dump(self.adapter.cfg, "Adapter Config")

    # ----------------- 내부 유틸 -----------------
    @staticmethod
    def _to_tensor(x, device):
        if x is None:
            raise RuntimeError("[Loss None] loss is None. Check LossComputer and model output.")
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _normalize_future_exo_from_cb(self, x, y, *, device) -> Optional[torch.Tensor]:
        """
        future_exo_cb가 무엇을 반환하든 최종 (B,H,E)로 정규화.
        """
        if self.future_exo_cb is None:
            return None

        B = x.size(0)
        H = y.size(1)
        t0 = 0

        exo = self.future_exo_cb(t0, H, device=device)
        if not torch.is_tensor(exo):
            raise TypeError(f"future_exo_cb must return torch.Tensor, got {type(exo)}")

        # squeeze 앞쪽 1차원
        while exo.dim() >= 3 and exo.size(0) == 1:
            exo = exo.squeeze(0)

        if exo.dim() == 2:
            exo = exo.unsqueeze(0)                   # (1,H,E)
        elif exo.dim() == 3:
            pass                                     # (B' or 1, H, E)
        elif exo.dim() == 4 and exo.size(0) == 1 and exo.size(1) == 1:
            exo = exo.squeeze(0).squeeze(0).unsqueeze(0)  # -> (1,H,E)
        else:
            raise RuntimeError(f"future_exo_cb returned unsupported shape={tuple(exo.shape)}")

        if exo.size(0) == 1 and B > 1:
            exo = exo.expand(B, -1, -1)
        elif exo.size(0) not in (1, B):
            raise RuntimeError(f"[EXO] batch mismatch: exo.shape[0]={exo.size(0)} vs B={B}")

        exo = exo.to(device)
        if not hasattr(self, "_logged_exo_shape"):
            print(f"[EXO-batch] exo normalized to shape={tuple(exo.shape)} (expect [B,H,E])")
            self._logged_exo_shape = True
        return exo

    def _resolve_future_exo(
        self,
        batch_future_exo: Optional[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        우선순위: 배치 제공 > cb 생성 > None
        """
        exo = None
        if batch_future_exo is not None:
            # (B,H,E) 가정. 빈 채널(E=0)이면 무시
            if torch.is_tensor(batch_future_exo) and batch_future_exo.ndim >= 2 and batch_future_exo.size(-1) > 0:
                exo = batch_future_exo.to(device)
        if exo is None:
            exo = self._normalize_future_exo_from_cb(x, y, device=device)

        if exo is not None:
            exo = torch.nan_to_num(exo, nan=0.0, posinf=1e6, neginf=-1e6)
            self._nan_stat("future_exo", exo)
        return exo

    def _nan_stat(self, name, t):
        if not torch.is_tensor(t):
            return
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        finite_mask = torch.isfinite(t)
        if finite_mask.any():
            try:
                mx = t[finite_mask].abs().max().item()
            except Exception:
                mx = t[finite_mask].to(torch.float32).abs().max().item()
        else:
            mx = float("inf")
        if has_nan or has_inf:
            print(f"[NaN-{name}] has_nan={has_nan} has_inf={has_inf} max|x|={mx}")

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[list], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        다양한 배치 시그니처를 표준화해 반환.
        Returns:
            x, y, part_ids, future_exo_cont, past_exo_cont, past_exo_cat
        """
        part_ids = None
        future_exo_cont = None
        past_exo_cont = None
        past_exo_cat = None

        if len(batch) == 6:
            x, y, part_ids, future_exo_cont, past_exo_cont, past_exo_cat = batch
        elif len(batch) == 5:
            x, y, part_ids, future_exo_cont, past_exo_cont = batch
        elif len(batch) == 3:
            x, y, part_ids = batch
        elif len(batch) == 2:
            x, y = batch
        else:
            raise RuntimeError(f"Unsupported batch tuple length: {len(batch)}")

        return x, y, part_ids, future_exo_cont, past_exo_cont, past_exo_cat

    def _forward_with_adapter(
        self,
        model,
        x,
        *,
        future_exo,
        past_exo_cont,
        past_exo_cat,
        part_ids,
        mode: str,
    ):
        """
        adapter.forward의 레거시/확장 시그니처를 모두 허용.
        """
        try:
            # 신 시그니처 (권장)
            return self.adapter.forward(
                model,
                x,
                future_exo=future_exo,
                past_exo_cont=past_exo_cont,
                past_exo_cat=past_exo_cat,
                part_ids=part_ids,
                mode=mode,
            )
        except TypeError:
            # 구 시그니처 (future_exo만 받는 경우)
            return self.adapter.forward(
                model,
                x,
                future_exo=future_exo,
                mode=mode,
            )

    # ----------------- 에폭 루프 -----------------
    def _run_epoch(self, model, loader, *, train: bool):
        device = self.cfg.device
        total = 0.0
        model.train() if train else model.eval()

        with torch.set_grad_enabled(train):
            for batch in loader:
                x, y, part_ids, fe_cont, pe_cont, pe_cat = self._unpack_batch(batch)
                x, y = x.to(device), y.to(device)

                # NaN/Inf 가드
                self._nan_stat("x(in)", x)
                self._nan_stat("y(in)", y)
                if pe_cont is not None: self._nan_stat("past_exo_cont", pe_cont)
                if pe_cat  is not None: self._nan_stat("past_exo_cat",  pe_cat)

                if train:
                    self.opt.zero_grad(set_to_none=True)

                # future_exo 결정 (배치 > cb)
                future_exo = self._resolve_future_exo(fe_cont, x, y, device=device)

                with autocast(
                    device_type=self.cfg.amp_device,
                    enabled=self.amp_enabled,
                    dtype=self.dtype if self.dtype is not None else "fp32",
                ):
                    pred = self._forward_with_adapter(
                        model,
                        x,
                        future_exo=future_exo,
                        past_exo_cont=(pe_cont.to(device) if torch.is_tensor(pe_cont) else None),
                        past_exo_cat=(pe_cat.to(device) if torch.is_tensor(pe_cat) else None),
                        part_ids=part_ids,
                        mode=("train" if train else "eval"),
                    )
                    self._nan_stat("pred", pred)

                    loss = self.loss_comp.compute(pred, y, is_val=(not train))
                    if self.extra_loss_fn is not None:
                        loss = loss + self.extra_loss_fn(x, pred, self.cfg)

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

                    # with torch.no_grad():
                    #     total_norm = 0.0
                    #     for name, p in model.named_parameters():
                    #         if p.grad is not None:
                    #             total_norm += p.grad.abs().mean().item()
                    #     print(f"[DEBUG] grad mean abs = {total_norm:.6f}")

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

        best_loss = float("inf")
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
                    x, y, part_ids, fe_cont, pe_cont, pe_cat = self._unpack_batch(batch)
                    x_val, y_val = x.to(device), y.to(device)
                    future_exo = self._resolve_future_exo(fe_cont, x_val, y_val, device=device)

                    if tta_steps > 0 and self.adapter.uses_tta():
                        loss = self.adapter.tta_adapt(model, x_val, y_val, steps=tta_steps)
                        if loss is None:
                            with autocast(
                                device_type=self.cfg.amp_device,
                                enabled=self.amp_enabled,
                                dtype=self.dtype if self.dtype is not None else "fp32",
                            ):
                                pred = self._forward_with_adapter(
                                    model,
                                    x_val,
                                    future_exo=future_exo,
                                    past_exo_cont=(pe_cont.to(device) if torch.is_tensor(pe_cont) else None),
                                    past_exo_cat=(pe_cat.to(device) if torch.is_tensor(pe_cat) else None),
                                    part_ids=part_ids,
                                    mode="eval",
                                )
                                loss = self.loss_comp.compute(pred, y_val, is_val=True)
                                if self.extra_loss_fn is not None:
                                    loss = loss + self.extra_loss_fn(x_val, pred, self.cfg)
                                loss = float(loss.detach())
                        val_total += loss
                    else:
                        with autocast(
                            device_type=self.cfg.amp_device,
                            enabled=self.amp_enabled,
                            dtype=self.dtype if self.dtype is not None else "fp32",
                        ):
                            pred = self._forward_with_adapter(
                                model,
                                x_val,
                                future_exo=future_exo,
                                past_exo_cont=(pe_cont.to(device) if torch.is_tensor(pe_cont) else None),
                                past_exo_cat=(pe_cat.to(device) if torch.is_tensor(pe_cat) else None),
                                part_ids=part_ids,
                                mode="eval",
                            )
                            vloss = self.loss_comp.compute(pred, y_val, is_val=True)
                            if self.extra_loss_fn is not None:
                                vloss = vloss + self.extra_loss_fn(x_val, pred, self.cfg)
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