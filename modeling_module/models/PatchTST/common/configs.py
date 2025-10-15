from typing import Optional, Literal, Tuple
from dataclasses import dataclass, field

from modeling_module.training.config import TrainingConfig, DecompositionConfig


# =========================
# Attention (Self-Attention) 설정
# =========================
@dataclass
class AttentionConfig(TrainingConfig):
    """
        PatchTST의 어텐션 관련 하이퍼파라미터 묶음.
        - 'full'은 표준 Multi-Head Self-Attention
        - 'probsparse'는 Informer류의 확률적 희소 어텐션(긴 시계열 가속 목적)
    """
    type: Literal['full', 'probsparse'] = 'full'    # 어텐션 유형
    n_heads: int = 16                               # multi head 개수 (d_model % n_heads == 0 권장)
    d_model: int = 128                              # token embedding dimension (model hidden dimension)
    d_k: Optional[int] = None                       # 각 헤드의 Query/Key dimension (기본 None이면 d_model // n_heads)
    d_v: Optional[int] = None                       # 각 헤드의 Value dimension (기본 None이면 d_model // n_heads)
    attn_dropout: float = 0.0                       # Attention Score (dropout)
    proj_dropout: float = 0.0                       # Attention Output Projection (dropout)
    causal: bool = True                             # Triangular Causal Mask 사용 (미래 차단) 여부
    residual_logits: bool = True                    # RealFormer Style Logit Residual 추가 여부
    output_attention: bool = False                  # 헤드별 어텐션 가중치 반환(디버깅/해석용) 여부
    factor: int = 5                                 # ProbSparse 전용 (키 샘플링 개수 스케일)
    lsa: bool = False                               # Local Self-Attn scale 학습 여부

# =========================
# Head 설정 (출력부)
# =========================
@dataclass
class HeadConfig:
    """
        모델 헤드 유형 및 출력부 하이퍼파라미터.
        - 'prediction' : 예측 전용 헤드 (시계열 포캐스팅)
        - 'flatten'    : 패치/채널 평탄화 후 선형 투사
        - 'pretrain'   : 자기지도 사전학습(재구성 등)용 헤드
    """
    type: Literal['prediction', 'flatten', 'pretrain'] = 'flatten'
    individual: bool = False
    head_dropout: float = 0.0
    y_range: Optional[Tuple[float, float]] = None


# =========================
# PatchTST 전역 설정
# =========================
@dataclass
class PatchTSTConfig(TrainingConfig):
    """
       PatchTST의 핵심 하이퍼파라미터 구성.

       [데이터/윈도우]
       - c_in       : 입력 채널 수(=모델 입력 변수 개수). 단변량은 1.
       - target_dim : 예측해야 하는 타깃 변수 개수(멀티타깃이면 >1). 일반적으로 target_dim <= c_in.
       - lookback   : 인코더가 보는 과거 길이(컨텍스트 윈도우).
       - horizon    : 예측할 미래 길이.
       - patch_len  : 한 패치가 커버하는 시간 길이(로컬 패턴 단위). 계절성이 뚜렷하면 주기와 정합(예: 12).
       - stride     : 패치 이동 간격(겹침 정도). 기본적으로 patch_len//2 권장(50% overlap).
       - padding_patch : 'end'면 말단에 패딩을 추가해 마지막 패치를 하나 더 확보(패치 손실 완화).

       [인코더/백본]
       - n_layers   : 인코더 레이어(블록) 개수.
       - d_model    : 모델 은닉 차원(패치 임베딩 차원).
       - d_ff       : FFN(포지션-와이즈) 내부 차원.
       - norm       : 정규화 유형('BatchNorm' 권장). 구현체에 따라 'LayerNorm' 등 선택 가능.
       - dropout    : 드롭아웃 비율(인코더 전반).
       - act        : 활성함수('relu'|'gelu' 등).
       - pre_norm   : True면 레이어 진입 전 정규화(Pre-LN). False면 Post-LN.
       - store_attn : True면 어텐션 가중치를 내부에 저장(디버깅용).
       - pe         : 위치임베딩 유형('zeros'|'learned' 등 프로젝트 구현에 따름).
       - learn_pe   : True면 위치임베딩을 학습.
       - revin      : True면 RevIN(분포 정규화) 사용.
       - affine     : RevIN의 affine 파라미터 사용 여부.
       - subtract_last : RevIN에서 마지막값 기준 보정 사용 여부.
       - verbose    : True면 디버그/로그 상세 출력.

       [서브설정]
       - attn       : 어텐션 설정(AttentionConfig)
       - head       : 헤드 설정(HeadConfig)
       """
    # ---------- 데이터/윈도우 ----------
    c_in: int = 1                       # 입력 채널 수
    target_dim: int = 1                 # 예측 타깃 채널 수
    patch_len: int = 8                  # 패치 길이(로컬 패턴 단위)
    stride: int = patch_len // 2        # 패치 간 이동 간격 (None 이면 patch_len//2 자동 설정)
    # lookback: int = 36                  # 과거 컨텍스트 길이 (encoder 입력 길이)
    # horizon: int = 48                   # 예측 길이 (decoder/헤드 출력 길이)
    padding_patch: Optional[str] = None # 'end' or None (끝 패딩 여부)

    # ---------- 인코더(백본) ----------
    n_layers: int = 3                   # 인코더 레이어 수
    d_model: int = 128                  # 은닉 차원(패치 임베딩 차원)
    d_ff: int = 256                     # FFN 내부 차원
    norm: str = 'BatchNorm'             # 정규화 유형('BatchNorm' 권장)
    dropout: float = 0.0                # Dropout ratio
    act: str = 'gelu'                   # Activation function
    pre_norm: bool = False              # Pre-LN 여부
    store_attn: bool = False            # 어텐션 가중치 저장
    pe: str = 'zeros'                   # Positional Embedding 유형
    learn_pe: bool = True               # Positional Embedding 학습 여부
    revin: bool = True                  # RevIN 사용 여부
    affine: bool = True                 # RevIN affine 사용 여부
    subtract_last: bool = False         # RevIN 마지막값 기준 보정
    verbose: bool = False               # 상세 로그
    head_type = 'regression'
    head_dropout: float = 0.
    individual = False



    # ---------- 서브 설정 ----------
    attn: AttentionConfig = field(default_factory=AttentionConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    decomp: DecompositionConfig = field(default_factory=DecompositionConfig)

# =========================
# 프리셋: 월간/주간 권장 윈도우
# =========================
@dataclass
class PatchTSTConfigMonthly(PatchTSTConfig):
    """
        월간 데이터 기본 프리셋:
        - lookback=36(3년), horizon=48(4년) 예시
        - patch_len은 월 주기(12)의 약수/배수를 권장(예: 12), stride는 patch_len//2 권장
    """
    # lookback: int = 36
    # horizon: int = 48
    pass

@dataclass
class PatchTSTConfigWeekly(PatchTSTConfig):
    """
        주간 데이터 기본 프리셋:
        - lookback=54, horizon=27 예시
        - patch_len은 8/12/16 등 주기와 데이터 특성에 맞춰 탐색, stride는 patch_len//2 권장
    """
    # lookback: int = 54
    # horizon: int = 27
    pass