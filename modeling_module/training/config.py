from dataclasses import dataclass, field, replace
from typing import Tuple, Optional, Literal
import torch




@dataclass
class DecompositionConfig:
    type: Optional[Literal["none", "ma", "stl"]] = "none"
    ma_window: int = 7
    stl_period: int = 12
    concat_mode: Literal["concat", "residual_only", "trend_only"] = "concat"

# ---------------- Spike-aware Loss Configs ---------------- #
@dataclass
class SpikeLossConfig:
    """
    스파이크 친화 손실을 켤지/어떻게 섞을지에 대한 설정.
    - enabled: True면 spike-friendly 조합 손실을 사용
    - strategy: 'mix' | 'direct'
        * 'mix'   : WeightedHuber + AsymmetricMSE를 (alpha, beta)로 섞음 (+ 옵션으로 baseline까지)
        * 'direct': 단일 함수형 손실(예: huber_asymmetric)을 그대로 사용
    - 아래 가중/임계값은 spikes나 과대예측 페널티를 제어
    """
    enabled: bool = True
    strategy: Literal['mix', 'direct'] = 'mix'

    # 공통 파라미터
    huber_delta: float = 2.0
    asym_up_weight: float = 2.0  # 과대예측( pred>y ) 가중
    asym_down_weight: float = 1.0

    # mix 전용(스파이크 위치 가중)
    mad_k: float = 3.5          # 스파이크 탐지 임계(k-MAD)
    w_spike: float = 6.0        # 스파이크 구간 가중
    w_norm: float = 1.0         # 일반 구간 가중
    w_cap = 12.0
    alpha_huber: float = 0.7    # WeightedHuber 비중
    beta_asym: float = 0.3      # AsymMSE 비중

    # baseline(기존 LossComputer)와 혼합할지
    mix_with_baseline: bool = False
    gamma_baseline: float = 0.2

    def with_enabled(self, flag: bool) -> "SpikeLossConfig":
        return replace(self, enabled=bool(flag))


@dataclass
class TrainingConfig:
    # ------------Loader------------
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' # For Mac
    lookback: int = 54
    horizon: int = 27

    # ------------Training------------
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    t_max: int = 10                 # CosineAnnealingLR
    patience: int = 50
    max_grad_norm: float = 30.0
    amp_device: str = 'cuda'        # 'cuda' | 'cpu'


    # ------------Loss-------------
    loss_mode: Literal['auto', 'point', 'quantile'] = 'auto'
    point_loss: Literal['mae','mse','huber','pinball','huber_asym', None] = 'mse'
    huber_delta: float = 5.0
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


    # ------------Exogenous Value --------------
    exo_dim = 2  # 미래 외생변수 차원(없으면 0)
    nonneg_head = False  # 수요 비음수 보장 (Softplus)


    # ------------Spike-friendly Loss-------------- #
    spike_loss: SpikeLossConfig = field(default_factory=SpikeLossConfig)

    lambda_hist_scale: float = 0.1
    lambda_hist_var: float = 0.03
    hist_window: int = 12

    anchor_last_k: int = 8  # 과거 K 스텝을 스케일/레벨 앵커로 사용
    anchor_weight: float = 0.05  # 앵커 손실 가중치

    def copy_with(self, **kwargs) -> "TrainingConfig":
        """dataclasses.replace 래퍼: 지정한 필드만 바꾼 사본을 반환"""
        return replace(self, **kwargs)

@dataclass
class StageConfig:
    epochs: int
    # 이 스테이지에서 덮어쓸 옵션들만 선택적으로 지정
    spike_enabled: Optional[bool] = None
    lr: Optional[float] = None
    use_horizon_decay: Optional[bool] = None
    tau_h: Optional[float] = None
    huber_delta: Optional[float] = None
    # (선택) 스파이크 가중·정의 강화
    w_spike: Optional[float] = None
    mad_k: Optional[float] = None
    asym_down_weight: Optional[float] = None

def apply_stage(base: TrainingConfig, stg: StageConfig) -> TrainingConfig:
    cfg = replace(base)
    if stg.spike_enabled is not None:
        cfg.spike_loss.enabled = stg.spike_enabled
    if stg.lr is not None:
        cfg.lr = stg.lr
    if stg.use_horizon_decay is not None:
        cfg.use_horizon_decay = stg.use_horizon_decay
    if stg.tau_h is not None:
        cfg.tau_h = stg.tau_h
    if stg.huber_delta is not None:
        cfg.huber_delta = stg.huber_delta
    if stg.w_spike is not None:
        cfg.spike_loss.w_spike = stg.w_spike
    if stg.mad_k is not None:
        cfg.spike_loss.mad_k = stg.mad_k
    if stg.asym_down_weight is not None:
        cfg.spike_loss.asym_down_weight = stg.asym_down_weight
    # 이 스테이지에서만 쓸 epoch 수
    cfg.epochs = stg.epochs
    return cfg