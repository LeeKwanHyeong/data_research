import torch
import torch.nn.functional as F
from typing import Tuple

from training.config import TrainingConfig
from utils.custom_loss_utils import (
    intermittent_weights_balanced,
    intermittent_point_loss,
    newsvendor_q_star,
    pinball_plain,
    pinball_loss_weighted_masked
)

class LossComputer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg

    def _q_star(self):
        if self.cfg.use_cost_q_star and self.cfg.point_loss == 'pinball':
            return newsvendor_q_star(self.cfg.Cu, self.cfg.Co)
        return self.cfg.q_star

    def compute(self, pred: torch.Tensor, y: torch.Tensor, * , is_val: bool) -> torch.Tensor:
        mode = self.cfg.loss_mode
        if mode == 'auto':
            mode = 'quantile' if pred.dim() == 3 else 'point'

        if mode == 'quantile':
            if is_val and not self.cfg.val_use_weight:
                return pinball_plain(pred, y, self.cfg.quantiles)

            weights = None
            if self.cfg.use_intermittent:
                weights = intermittent_weights_balanced(
                    y, alpha_zero = self.cfg.alpha_zero, alpha_pos = self.cfg.alpha_pos,
                    gamma_run = self.cfg.gamma_run, clip_run = 0.5
                )
            return pinball_loss_weighted_masked(pred, y, self.cfg.quantiles, weights)

        # point
        if is_val and not self.cfg.val_use_weight:
            if self.cfg.point_loss == 'mae':
                return (pred - y).abs().mean()
            if self.cfg.point_loss == 'mse':
                return F.mse_loss(pred, y)
            if self.cfg.point_loss == 'huber':
                return F.huber_loss(pred, y, delta = self.cfg.huber_delta)
            if self.cfg.point_loss == 'pinball':
                q = self._q_star()
                diff = y - pred
                return torch.maximum(q * diff, (q-1.0)*diff).mean()

        return intermittent_point_loss(
            pred, y,
            mode = self.cfg.point_loss,
            tau = self._q_star(),
            delta = self.cfg.huber_delta,
            alpha_zero = self.cfg.alpha_zero if self.cfg.use_intermittent else 0.0,
            alpha_pos = self.cfg.alpha_pos,
            gamma_run = self.cfg.gamma_run,
            cap = self.cfg.cap,
            use_horizon_decay = self.cfg.use_horizon_decay,
            tau_h = self.cfg.tau_h
        )
