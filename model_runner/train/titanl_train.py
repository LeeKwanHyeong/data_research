import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch import GradScaler, amp

from models.Titan.Titans import TitanConfigMonthly, TestTimeMemoryManager
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

