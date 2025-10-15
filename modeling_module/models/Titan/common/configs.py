from dataclasses import dataclass

from modeling_module.training.config import TrainingConfig


@dataclass
class TitanConfig(TrainingConfig):
    batch_size: int = 128               # 한 번의 학습/추론에서 처리할 시계열 샘플 개수 (배치 크기)
    lookback: int = 36                  # 입력 시계열의 과거 타임스텝 길이 (모델 입력 시 고려하는 시점 수)
    horizon: int = 48                   # 예측할 미래 타임스텝 길이 (출력 시점 수)
    input_dim: int = 1                  # 각 타임스텝에서의 피처(feature) 개수 (예: 단변량=1, 다변량>1)
    d_model: int = 64                   # 모델 내부 임베딩 차원 (토큰/패치가 변환되는 은닉 공간의 차원)
    n_layers: int = 2                   # Transformer 또는 Titan 블록의 층(layer) 개수
    n_heads: int = 4                    # Multi-Head Attention에서 병렬로 사용하는 어텐션 헤드 수
    d_ff: int = 256                     # 피드포워드 네트워크(FFN) 내부 차원 (확장된 은닉층 크기)
    contextual_mem_size: int = 64       # MAC(Memory-as-Context)에서 사용하는 컨텍스트 메모리 슬롯 개수
    persistent_mem_size: int = 16       # MAC에서 사용하는 영구 메모리(persistent memory) 슬롯 개수

    # --------------- Decoder -----------------
    n_dec_layers = 1                    # 1~2 권장
    dec_dropout = 0.1

    @property
    def output_horizon(self) -> int:
        return self.horizon

@dataclass
class TitanConfigMonthly(TitanConfig):
    lookback: int = 36
    horizon: int = 48

@dataclass
class TitanConfigWeekly(TitanConfig):
    lookback: int = 54
    horizon: int = 27