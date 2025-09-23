import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch import GradScaler, amp

from models.Titan.Titans import TestTimeMemoryManager
from models.Titan.common.configs import TitanConfigMonthly


# import mlflow
# import mlflow.pytorch

# mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

class TitanTrain:
    def __init__(self):
        self.titan_config = TitanConfigMonthly()
        # mlflow.set_tracking_uri("file:///Users/igwanhyeong/mlruns")  # 또는 원하는 로컬 경로
        # mlflow.set_tracking_uri("http://192.168.219.111:5000")
        # mlflow.set_experiment("TitanForecasting")
        # mlflow.set_tracking_uri("http://host.docker.internal:5000")
        # mlflow.set_experiment("TitanForecasting")


    def base_train(self, model, train_loader, val_loader, epochs, lr = 1e-3, device = 'cuda'):
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
        loss_fn = nn.MSELoss()
        scaler = GradScaler()

        max_grad_norm = 30
        patience = 50
        best_loss = float('inf')
        counter = 0
        best_model_state = copy.deepcopy(model.state_dict())


        train_losses, val_losses = [], []
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                with amp.autocast('cuda'):
                    pred = model(x_batch)
                    loss = loss_fn(pred, y_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    with amp.autocast('cuda'):
                        pred = model(x_val)
                        val_loss += loss_fn(pred, y_val).item()

            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Scheduler
            scheduler.step()

            # Early Stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    model.load_state_dict(best_model_state)
                    break

            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f}")
        model.load_state_dict(best_model_state)
        return model, (train_losses, val_losses)

    def base_with_custom_loss(self,
                              model,
                              train_loader,
                              val_loader,
                              *,
                              epochs: int = 100,
                              lr: float = 1e-3,
                              device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                              # --- 손실 설정 ---
                              loss_type: str = "point",     # "point" | "quantile"
                              loss_params: dict | None = None,
                              # --- 학습 일반 ---
                              patience: int = 30,
                              max_grad_norm: float = 30.0,
                              t_max: int = 10):
        """
        loss_type:
          - "point"    : 점추정 출력 [B,H] (Titan/LMMModel 기본) 에 적용
                         -> intermittent_point_loss 사용
                         -> loss_params 예: {
                               "mode": "huber",    # 'mae'|'mse'|'huber'|'pinball'
                               "delta": 1.0,       # huber
                               "tau": 0.5,         # pinball q
                               "alpha_zero": 1.2, "alpha_pos": 1.0, "gamma_run": 0.6, "cap": None,
                               "use_horizon_decay": False, "tau_h": 24.0
                            }

          - "quantile" : 멀티-분위수 출력 [B,Q,H] 에 적용
                         -> intermittent_pinball_loss 사용
                         -> loss_params 예: {
                               "quantiles": (0.1, 0.5, 0.9),
                               "alpha_zero": 1.2, "alpha_pos": 1.0, "gamma_run": 0.6, "cap": None,
                               "use_horizon_decay": False, "tau_h": 24.0
                            }

        반환: (best_model, (train_losses, val_losses))
        """
        loss_params = loss_params or {}
        model.to(device)

        # -----------------------------------
        # 손실 계산 클로저
        # -----------------------------------
        def _compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if loss_type.lower() == "point":
                # pred: [B,H]
                if pred.dim() != 2:
                    raise ValueError(f"[point] expects pred [B,H], got {tuple(pred.shape)}")
                return intermittent_point_loss(
                    y_hat=pred, target=target,
                    mode=str(loss_params.get("mode", "mae")),
                    tau=float(loss_params.get("tau", 0.5)),
                    delta=float(loss_params.get("delta", 1.0)),
                    alpha_zero=float(loss_params.get("alpha_zero", 1.2)),
                    alpha_pos=float(loss_params.get("alpha_pos", 1.0)),
                    gamma_run=float(loss_params.get("gamma_run", 0.6)),
                    cap=loss_params.get("cap", None),
                    use_horizon_decay=bool(loss_params.get("use_horizon_decay", False)),
                    tau_h=float(loss_params.get("tau_h", 24.0)),
                )

            elif loss_type.lower() == "quantile":
                # pred: [B,Q,H] (없으면 [B, Q*H] -> [B,Q,H]로 복구 시도)
                quantiles = tuple(loss_params.get("quantiles", (0.1, 0.5, 0.9)))
                if pred.dim() == 2:
                    # [B, Q*H]라고 가정하고 복원
                    B, QH = pred.shape
                    H = target.shape[1]
                    Q = len(quantiles)
                    if Q * H != QH:
                        raise ValueError(f"[quantile] cannot infer [Q,H]: pred=[B,{QH}] but Q={Q}, H={H} → {Q*H} mismatch")
                    pred = pred.view(B, Q, H)
                elif pred.dim() != 3:
                    raise ValueError(f"[quantile] expects pred [B,Q,H], got {tuple(pred.shape)}")

                return intermittent_pinball_loss(
                    pred_q=pred, target=target,
                    quantiles=quantiles,
                    alpha_zero=float(loss_params.get("alpha_zero", 1.2)),
                    alpha_pos=float(loss_params.get("alpha_pos", 1.0)),
                    gamma_run=float(loss_params.get("gamma_run", 0.6)),
                    cap=loss_params.get("cap", None),
                    use_horizon_decay=bool(loss_params.get("use_horizon_decay", False)),
                    tau_h=float(loss_params.get("tau_h", 24.0)),
                )

            else:
                raise ValueError("loss_type must be one of {'point','quantile'}")

        # -----------------------------------
        # Optim/Sched/AMP
        # -----------------------------------
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        scaler = GradScaler()

        best_loss = float('inf')
        counter = 0
        best_state = copy.deepcopy(model.state_dict())
        train_curve, val_curve = [], []

        for epoch in range(epochs):
            # ===== Train =====
            model.train()
            total = 0.0
            for batch in train_loader:
                if len(batch) == 3:
                    x_batch, y_batch, _ = batch
                else:
                    x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad(set_to_none=True)
                with amp.autocast('cuda'):
                    pred = model(x_batch)
                    loss = _compute_loss(pred, y_batch)

                if torch.isnan(loss):
                    print(f"[Warn] NaN loss at epoch {epoch}, skip step")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss)

            avg_train = total / max(1, len(train_loader))
            train_curve.append(avg_train)

            # ===== Val =====
            model.eval()
            vtotal = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        x_val, y_val, _ = batch
                    else:
                        x_val, y_val = batch
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    with amp.autocast('cuda'):
                        pred = model(x_val)
                        vloss = _compute_loss(pred, y_val)
                        vtotal += float(vloss)

            avg_val = vtotal / max(1, len(val_loader))
            val_curve.append(avg_val)

            # Scheduler / EarlyStopping
            scheduler.step()
            if avg_val < best_loss:
                best_loss = avg_val
                counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early Stopping at epoch {epoch+1}")
                    break

            cur_lr = scheduler.get_last_lr()[0]
            print(f"[{loss_type}] Epoch {epoch+1:03d}/{epochs} | LR {cur_lr:.6f} | Train {avg_train:.5f} | Val {avg_val:.5f}")

        model.load_state_dict(best_state)
        return model, (train_curve, val_curve)


    def train_model_with_tta(self,
                             model,
                             train_loader,
                             val_loader,
                             epochs = 100,
                             lr = 1e-3,
                             device = 'cuda' if torch.cuda.is_available() else 'cpu',
                             tta_steps = 0,
                             patience = 30,
                             max_grad_norm = 30,
                             t_max = 10):
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = t_max)
        loss_fn = nn.MSELoss()
        scaler = GradScaler()
        tta_manager = TestTimeMemoryManager(model, lr = lr)

        best_loss = float('inf')
        counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            # Training Loop
            for batch in train_loader:
                x_batch, y_batch, _ = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                with amp.autocast('cuda'):
                    pred = model(x_batch)
                    loss = loss_fn(pred, y_batch)

                if torch.isnan(loss):
                    print(f"[Error] Loss is NaN at epoch {epoch}, skipping step")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation Loop (with optional TTA)
            model.eval()
            val_loss = 0

            for batch in val_loader:
                x_val, y_val, _ = batch
                x_val, y_val = x_val.to(device), y_val.to(device)

                if tta_steps > 0:
                    # Test-Time Adaptation 수행
                    tta_manager.add_context(x_val)  # memory 업데이트
                    loss = tta_manager.adapt(x_val, y_val, steps=tta_steps)
                    val_loss += loss
                else:
                    with torch.no_grad(), amp.autocast('cuda'):
                        pred = model(x_val)
                        loss = loss_fn(pred, y_val)
                        val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # LR Scheduler Step
            scheduler.step()

            # Early Stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                counter = 0
                best_model_state = copy.deepcopy(model.state_dict())

            else:
                counter += 1
                if counter >= patience:
                    print(f"Early Stopping at epoch {epoch + 1}")
                    model.load_state_dict(best_model_state)
                    break

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f}"
                f"| Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )
        model.load_state_dict(best_model_state)
        return model, (train_losses, val_losses)

