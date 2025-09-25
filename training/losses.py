import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from training.config import TrainingConfig
from utils.custom_loss_utils import (
    intermittent_weights_balanced,      # 간헐 수요(intermittent demand)에서 0-run/양의 구간을 균형 가중
    intermittent_point_loss,            # 간헐 수요 대응 포인트 손실 (MAE/MSE/Huber/Pinball 변형 + 시점/런 길이 가중 포함)
    newsvendor_q_star,                  # 뉴스벤더 문제 기반 최적 분위수 q* = Cu / (Cu + Co)
    pinball_plain,                      # 가중치/마스킹 없는 표준 핀볼(분위수) 손실
    pinball_loss_weighted_masked        # 샘플/시점 가중 및 마스크 적용 가능한 핀볼 손실
)

class LossComputer:
    """
    Integrated Loss Computer
    - 'quantile' Mode: 예측이 (B, Q, H)처럼 다중 분위수(quantiles)를 출력하는 모델용 -> Pinball 계열 손실
    - 'point'    Mode: 예측이 (B, Q) 또는 (B, C, H) 등 단일 포인트 예측 모델용 -> MAE/MSE/Huber/Pinball(Single q*) 사용
    - 'auto'     Mode: pred.dim() == 3이면, 'quantile', 아니면 'point'로 자동 결정

    간헐 수요(0이 자주 등장) 시에는 run-length/zero/positive 구간에 대한
    가중치를 부여하여 학습의 방향성을 제어.
    검증 단계(is_val = True)에서는 cfg.val_use_weights에 따라 가중치 적용을 비활성화 할 수 있음.
    """
    def __init__(self, cfg: TrainingConfig):
        """
        cfg: Training 구성 객체
            - loss_mode: 'auto' | 'quantile' | 'point'
            - point_loss: 'mae' | 'mse' | 'huber' | 'pinball'
            - quantiles: List[float], Ex: [0.1, 0.5, 0.9]
            - use_intermittent: intermittent demand weight usage
            - huber_delta: Huber loss delta
            - cap: loss upper bound(cap) 등 안전장치(using inside of intermittent_point_loss)
            - use_horizon_decay, tau_h: horizon(lead time)별 decay weight setting
            - use_cost_q_star, Cu, Co, q_star 뉴스벤더 비용 기반 q* 사용 여부 및 비용/기본 분위수
        """
        self.cfg = cfg

    def _q_star(self):
        """
        point 손실이 'pinball'일 때 사용할 단일 분위수 q*를 결정
        - use_cost_q_star = True이고 point_loss == 'pinball'이면 뉴스벤더 q* = Cu / (Cu + Co)
        - else cfg.q_star 사용
        """
        if self.cfg.use_cost_q_star and self.cfg.point_loss == 'pinball':
            return newsvendor_q_star(self.cfg.Cu, self.cfg.Co)
        return self.cfg.q_star

    def compute(self, pred: torch.Tensor, y: torch.Tensor, * , is_val: bool) -> torch.Tensor:
        """
        주어진 예측과 정답에 대해 손실 계산
        입력 텐서 형태 가이드:
        - Quantile Forecasting: pred.shape == (B, Q, H) / y.shape == (B, 1, H) or (B, H)
        - Point    Forecasting: pred.shape == (B, C, H) / y.shape == pred or (B, H)

        pred: Model Forecasted Tensor
        y: Target Tensor
        is_val: 검증 단계 여부.
        is_val = True and cfg.val_use_weights = False then using 'plain loss' without sample/time weight

        Returns:
            torch.Tensor: Scalar Loss (shape == [])
        """
        # select Loss mode
        mode = self.cfg.loss_mode
        if mode == 'auto':
            mode = 'quantile' if pred.dim() == 3 else 'point'

        if mode == 'quantile':
            if is_val and not self.cfg.val_use_weights:
                return pinball_plain(pred, y, self.cfg.quantiles)

            weights: Optional[torch.Tensor] = None
            if self.cfg.use_intermittent:
                # y의 0-run/양의 구간 균형을 맞추는 가중치 생성
                # alpha_zero: 0 수요 가중, alpha_pos: 양의 수요 가중
                # gamma_run: run-length sensitivity, clip_run: prevent weight explode
                weights = intermittent_weights_balanced(
                    y,
                    alpha_zero = self.cfg.alpha_zero,
                    alpha_pos = self.cfg.alpha_pos,
                    gamma_run = self.cfg.gamma_run,
                    clip_run = 0.5
                )
            return pinball_loss_weighted_masked(pred, y, self.cfg.quantiles, weights)

        # point
        if is_val and not self.cfg.val_use_weights:
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
