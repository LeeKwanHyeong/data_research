from typing import Optional, Literal, Tuple
from dataclasses import dataclass, field

from modeling_module.training.config import TrainingConfig, DecompositionConfig


# =========================
# PatchMixer 전용 설정
# =========================
@dataclass
class PatchMixerConfig(TrainingConfig):
    """
    PatchMixer 공통 설정 (TrainingConfig 상속)
    - enc_in                 : 입력 채널 수 (단변량 1 권장)
    - d_model, e_layers      : 백본 용량
    - patch_len, stride      : 패치화 하이퍼파라미터
    - expander_f_out         : TemporalExpander 출력 특성 차원
    - expander_*             : 계절/주기 관련 하이퍼
    - exo_dim                : 미래 외생 변수 차원
    - use_part_embedding     : 파트 임베딩 사용 여부
    - part_vocab_size        : 파트 개수(임베딩 테이블 크기)
    - part_embed_dim         : 파트 임베딩 차원
    - final_nonneg           : 추론 시 음수 clamp
    - use_eol_prior          : EOL(수명 진행) 편향 적용
    - eol_feature_index      : future_exo에서 EOL proxy가 있는 feature index
    - exo_is_normalized_default : 외생치(정규화 공간)에서 가산할지의 기본값
    - head_hidden            : Base/Quantile head 내부 hidden 크기
    - (멀티스케일용) patch_cfgs, per_branch_dim, fused_dim, fusion, head_dropout 등
    """
    # ---------- 입력 ----------
    enc_in: int = 1

    # ---------- 백본 ----------
    d_model: int = 64
    e_layers: int = 3
    patch_len: int = 12
    stride: int = 8
    dropout: float = 0.1
    mixer_kernel_size: int = 5

    # ---------- Expander ----------
    f_out: int = 128
    expander_season_period: int = 52
    expander_n_harmonics: int = 8
    expander_max_harmonics: int = 16  # 호환용

    # ---------- 외생 ----------
    exo_dim: int = 0
    exo_is_normalized_default: bool = True

    # ---------- 파트 임베딩 ----------
    use_part_embedding: bool = False
    part_vocab_size: int = 0
    part_embed_dim: int = 16

    # ---------- 출력/제약 ----------
    final_nonneg: bool = True
    head_hidden: int = 128

    # ---------- EOL prior ----------
    use_eol_prior: bool = False
    eol_feature_index: int = 0  # future_exo[:, :, idx]

    # ---------- (멀티스케일 백본용) ----------
    patch_cfgs: tuple = ()
    per_branch_dim: int = 64
    fused_dim: int = 128
    fusion: str = "concat"
    head_dropout: float = 0.0

    # 호환성 (일부 구현에서 이름만 다르게 참조)
    season_period: int = 52
    max_harmonics: int = 16

    # RevIN 등에서 필요할 수 있어 유지
    decomp: DecompositionConfig = field(default_factory=DecompositionConfig)


# =========================
# 프리셋: 월간/주간
# =========================
@dataclass
class PatchMixerConfigMonthly(PatchMixerConfig):
    """
    월간 데이터 기본 프리셋:
    - lookback/horizon은 호출부에서 지정
    - month sin/cos 외생을 주로 사용(exo_dim=2 권장)
    """
    expander_season_period: int = 12
    expander_n_harmonics: int = 6


@dataclass
class PatchMixerConfigWeekly(PatchMixerConfig):
    """
    주간 데이터 기본 프리셋:
    - lookback/horizon은 호출부에서 지정
    - week-of-year sin/cos 외생(exo_dim=2) + 필요시 sequence(=EOL proxy)
    """
    expander_season_period: int = 52
    expander_n_harmonics: int = 8
