from dataclasses import dataclass
from typing import Literal

from modeling_module.training.config import TrainingConfig


@dataclass
class PatchMixerConfig(TrainingConfig):
    batch_size: int = 128
    d_model: int = 16
    e_layers: int = 2
    patch_len: int = 12
    stride: int = 8
    head_dropout: float = 0.05
    enc_in: int = 1
    mixer_kernel_size: int = 8

    # Multi-scale backbone/head 공통
    patch_cfgs = ((4, 2, 5), (8, 4, 7), (12, 6, 9))
    fused_dim = 256
    per_branch_dim: int = 128
    fusion: str = 'concat'

    # Temporal Expander 옵션
    date_type: Literal['M', 'W'] = 'M'                      # 'M' 월간, 'W' 주간
    expander_season_period: int = 12                                 # TemporalExpander의 사인/코사인 주기
    expander_max_harmonics: int = 8                                  # TemporalExpander에서 쓰는 최대 하모닉 수
    expander_n_harmonics: int = 4                                    # DecompositionQuantileHead용 Fourier K
    expander_f_out: int = 128                                        # Expander 출력 차원

    # Exogenous (현재 단계: calendar_sin_cos만 사용)
    use_calendar_exo: bool = True
    exo_dim: int = 2                             # sin, cos 두 채널

    @property
    def output_horizon(self) -> int:
        return self.horizon


@dataclass
class PatchMixerConfigMonthly(PatchMixerConfig):
    # 월간 기본값
    date_type: str = 'M'
    # 데이터 특성에 맞게 권장 기본값 예시
    lookback: int = 36
    horizon: int = 48
    expander_season_period: int = 12
    expander_max_harmonics: int = 8
    expander_n_harmonics: int = 6   # 월간은 낮은 K 권장(지나친 고주파 억제)


@dataclass
class PatchMixerConfigWeekly(PatchMixerConfig):
    # 주간 기본값
    date_type: str = 'W'
    lookback: int = 54
    horizon: int = 27
    expander_season_period: int = 52
    expander_max_harmonics: int = 16
    expander_n_harmonics: int = 8   # 주간은 계절 성분이 풍부해 K를 조금 더 허용
