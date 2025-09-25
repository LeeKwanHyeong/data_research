from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal

LossMode = Literal['auto', 'point', 'quantile']


@dataclass
class DecompositionConfig:
    type: Optional[Literal["none", "ma", "stl"]] = "none"
    ma_window: int = 7
    stl_period: int = 12
    concat_mode: Literal["concat", "residual_only", "trend_only"] = "concat"


@dataclass
class TrainingConfig:
    # ------------Loader------------
    device: str = 'cuda'
    lookback: int = 36
    horizon: int = 48

    # ------------Training------------
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    t_max: int = 10                 # CosineAnnealingLR
    patience: int = 50
    max_grad_norm: float = 30.0
    amp_device: str = 'cuda'        # 'cuda' | 'cpu'
    # ------------Loss-------------
    loss_mode: LossMode = 'auto'
    point_loss: Literal['mae', 'mse', 'huber', 'pinball'] = 'mse'
    huber_delta: float = 2.0
    q_star: float = 0.5             # point=pinball
    use_cost_q_star: bool = False
    Cu: float = 1.0; Co: float = 1.0 # newsvendor
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    # ------------Intermittent/Horizon Weight-------------
    use_intermittent: bool = True
    alpha_zero: float = 1.2
    alpha_pos: float = 1.0
    gamma_run: float = 0.6
    cap: Optional[float] = None
    use_horizon_decay: bool = False
    tau_h: float = 24.0
    # ------------Validation Weight-------------
    val_use_weights: bool = False    # 공정평가면 False
