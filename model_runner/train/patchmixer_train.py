import copy
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import GradScaler, amp

# import mlflow
# import mlflow.pytorch

# 간헐수요 가중/손실 유틸
from utils.losses import (
    intermittent_pinball_loss,
    intermittent_point_loss,
    newsvendor_q_star
)

from models.Titans import TestTimeMemoryManager

class PatchMixerTrain:
    """
    TitanTrain과 동일한 UX + PatchMixer 특화 가능:
        - pred.shape 가 (B, H)면: 점추정 모드 -> MAE/MSE/Huber/single pinball (q*) 지원
        - pred.shape 가 (B, Q, H)면: 분위수 모두 -> 간헐 가중 Pinball 지원
        - 동적 패칭(Offset) 사용 시: patcher.last_reg_loss() 자동 합산
    """

    def __init__(self):
        # mlflow.set_tracking_uri("http://host.docker.internal:5000")
        # mlflow.set_experiment("PatchMixerForecasting")
        pass

    # -----------------------------
    # 내부 헬퍼
    # -----------------------------
    @staticmethod
    def _forward_model(model: nn.Module, x_batch):
        """x_batch가 (ts, feat) tuple unback 호출, 아니면 그대로."""
        if isinstance(x_batch, (tuple, list)):
            return model(*x_batch)
        return model(x_batch)

    @staticmethod
    def _maybe_offset_reg(model: nn.Module) -> torch.Tensor:
        """
        DynamicOffsetPatcher를 사용할 때 정규화 항을 자동으로 수집
        model.backbone.patcher.last_reg_loss()가 있으면 더함.
        """
        try:
            patcher = getattr(getattr(model, 'backbone'), 'patcher', None)
            if patcher is not None and hasattr(patcher, 'last_reg_loss'):
                return patcher.last_reg_loss()
        except Exception:
            pass

        return torch.tensor(0.0, device = next(model.parameters()).device)

    # -----------------------------
    # 손실 계산 (학습/검증 공용)
    # -----------------------------
    def _compute_loss(self,
                      pred: torch.Tensor,       # (B, H) or (B, Q, H)
                      y:torch.Tensor,           # (B, H)
                      *,
                      mode: str = 'auto',       # 'auto' | 'point' | 'quantile'
                      # point mode
                      point_loss:str = 'mse',   # 'mae' | 'mse' | 'huber' | 'pinball'
                      q_star: float = 0.5,      # point == 'pinball'일 때 사용
                      huber_delta: float = 1.0,
                      # quantile mode
                      quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
                      # intermittent weight option
                      use_intermittent: bool = True,
                      alpha_zero: float = 1.2,
                      alpha_pos: float = 1.0,
                      gamma_run: float = 0.6,
                      cap: float | None = None,
                      # horizon decay
                      use_horizon_decay: bool = False,
                      tau_h: float = 24.0,
                      # 검증 시 가중을 끌지 여부
                      val_use_weights: bool = False,
                      is_val: bool = False,
                      ) -> torch.Tensor:
        """
        - mode = 'auto': pred.ndim으로 자동 결정
        - 학습 시 (use_intermittent = True) 간헐 가중 적용, 검증에서는 기본 False 권장 (val_use_weights)
        """
        if mode == 'auto':
            mode = 'quantile' if pred.dim() == 3 else 'point'

        if mode == 'quantile':
            # pred: (B, Q, H)
            if is_val and not val_use_weights:
                # 검증은 가중 없이 공정 평가(권장)
                return intermittent_pinball_loss(
                    pred, y, quantiles = quantiles, alpha_zero = alpha_zero,
                    alpha_pos = alpha_pos, gamma_run = gamma_run, cap = cap,
                    use_horizon_decay = False # 검증에서는 일반적으로 horizon 가중 비적용
                ) if use_intermittent and val_use_weights else \
                    intermittent_pinball_loss(
                    pred, y, quantiles = quantiles, alpha_zero = alpha_zero,
                        alpha_pos = alpha_pos, gamma_run = gamma_run,
                    cap = cap, use_horizon_decay = False
                ) * 0 + \
                intermittent_pinball_loss(
                    pred, y, quantiles = quantiles,
                    alpha_zero = alpha_zero, alpha_pos = alpha_pos, gamma_run = gamma_run,
                    cap = cap, use_horizon_decay = False
                ) # 형태 유지용 (간단히 기본 핀볼 한 번 더 부르지 않고 싶다면 별도 비가중 구현 가능)
            else:
                return intermittent_pinball_loss(
                    pred, y, quantiles = quantiles,
                    alpha_zero = alpha_zero if use_intermittent else 0.0, # 간헐 비활성 시 효과 거의 없음
                    alpha_pos = alpha_pos, gamma_run = gamma_run, cap = cap,
                    use_horizon_decay = use_horizon_decay, tau_h = tau_h
                )
        else: # point
            if is_val and not val_use_weights:
                # 검증은 비가중 손실(공정 평가)
                if point_loss == 'mae':
                    return (pred - y).abs().mean()
                if point_loss == 'mse':
                    return nn.functional.mse_loss(pred, y)
                if point_loss == 'huber':
                    return nn.functional.huber_loss(pred, y, delta = huber_delta)
                if point_loss == 'pinball':
                    diff = y - pred
                    e = torch.maximum(q_star * diff, (q_star - 1.0) * diff)
                    return e.mean()

            return intermittent_point_loss(
                pred, y,
                mode = point_loss,
                tau = q_star,
                delta = huber_delta,
                alpha_zero = alpha_zero if use_intermittent else 0.0,
                alpha_pos = alpha_pos,
                gamma_run = gamma_run,
                cap = cap,
                use_horizon_decay = use_horizon_decay,
                tau_h = tau_h
            )

    def base_train(self,
                   model: nn.Module,
                   train_loader: Iterable,
                   val_loader: Iterable,
                   epochs: int,
                   lr: float = 1e-3,
                   device: str = 'cuda',
                   # loss
                   loss_mode: str = 'auto',     # 'auto'|'point'|'quantile'
                   point_loss: str = 'mse',     # point일 때: 'mae'|'mse'|'huber'|'pinball'
                   quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
                   # cost based q* (using for pinball)
                   use_cost_qstar: bool = False,
                   Cu: float = 1.0,
                   Co: float = 1.0,
                   q_star_fallback: float = 0.5,
                   # intermittent/horizon weighted
                   use_intermittent: bool = True,
                   alpha_zero: float = 1.2,
                   alpha_pos: float = 1.0,
                   gamma_run: float = 0.6,
                   cap: float | None = None,
                   use_horizon_decay: bool = False,
                   tau_h: float = 24.0,
                   # 기타
                   patience: int = 50,
                   max_grad_norm: float = 30.0
                   ):
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
        scaler = GradScaler()

        # if point pinball
        q_star = newsvendor_q_star(Cu, Co) if (use_cost_qstar and point_loss == 'pinball') else q_star_fallback

        best_loss = float('inf')
        counter = 0
        best_state = copy.deepcopy(model.state_dict())

        train_losses, val_losses = [], []
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                # batch unpack (x, y) or (x, y, meta)
                if len(batch) == 2:
                    x_batch, y_batch = batch
                else:
                    x_batch, y_batch, _ = batch

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad(set_to_none = True)
                with amp.autocast('cuda'):
                    pred = self._forward_model(model, x_batch)
                    loss = self._compute_loss(
                        pred, y_batch,
                        mode = loss_mode,
                        point_loss = point_loss,
                        q_star = q_star,
                        quantiles = quantiles,
                        use_intermittent = use_intermittent,
                        alpha_zero = alpha_zero, alpha_pos = alpha_pos,
                        gamma_run = gamma_run,
                        use_horizon_decay = use_horizon_decay,
                        tau_h = tau_h, is_val = False,
                    )
                    # 동적 오프셋 패칭 정규화 항
                    loss = loss + self._maybe_offset_reg(model)
                if torch.isnan(loss):
                    print(f"[Warn] NaN loss at epoch {epoch + 1}, skip step")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach())

            avg_train_loss = total_loss / max(1, len(train_loader))

            # -------------------------Validation----------------------------
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 2:
                        x_val, y_val = batch
                    else:
                        x_val, y_val, _ = batch

                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    with amp.autocast('cuda'):
                        pred_val = self._forward_model(model, x_val)
                        vloss = self._compute_loss(
                            pred_val, y_val,
                            mode = loss_mode,
                            point_loss = point_loss,
                            q_star = q_star,
                            quantiles = quantiles,
                            # 검증은 공정 평가(비가중) 권장
                            use_intermittent = use_intermittent,
                            alpha_zero = alpha_zero, alpha_pos = alpha_pos,
                            gamma_run = gamma_run,
                            use_horizon_decay = False, # 일반적으로 비활성
                            is_val = True,
                            val_use_weights = False,
                        )
                    val_total += float(vloss.detach())

            avg_val_loss = val_total / max(1, len(val_loader))
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step()

            # Early Stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    model.load_state_dict(best_state)
                    break

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{epochs} | LR {current_lr:.6f}"
                f"| Train {avg_val_loss:.4f} | Val {avg_val_loss:.4f}"
            )

        model.load_state_dict(best_state)
        return model, train_losses, val_losses