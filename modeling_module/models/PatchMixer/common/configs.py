from dataclasses import dataclass

from modeling_module.training.config import TrainingConfig


@dataclass
class PatchMixerConfig(TrainingConfig):
    batch_size: int = 128           # 한 번의 학습/추론에서 처리할 시계열 샘플 개수 (배치 크기)
    # lookback: int = 36            # 입력 시계열의 과거 타임스텝 길이 (모델 입력 시 고려하는 시점 수)
    # horizon: int = 48             # 예측할 미래 타임스텝 길이 (출력 시점 수)
    d_model: int = 16               # 모델 내부 임베딩 차원 (토큰/패치가 변환되는 은닉 공간의 차원)
    e_layers: int = 2               # Transformer 또는 PatchMixer 블록의 층(layer) 개수
    patch_len: int = 12             # Patch Length
    stride: int = 8
    head_dropout: float = 0.1       # Residual Connection을 위한 dropout
    enc_in: int = 1
    mixer_kernel_size: int = 8
    @property
    def output_horizon(self) -> int:
        return self.horizon

@dataclass
class PatchMixerConfigMonthly(PatchMixerConfig):
    # lookback: int = 36
    # horizon: int = 48
    pass

@dataclass
class PatchMixerConfigWeekly(PatchMixerConfig):
    # lookback: int = 54
    # horizon: int = 27
    pass