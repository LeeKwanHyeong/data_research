from typing import Optional, Literal, Tuple
from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    type: Literal['full', 'probsparse'] = 'full'
    n_heads: int = 16
    d_model: int = 128
    d_k: Optional[int] = None
    d_v: Optional[int] = None
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    causal: bool = True                 # Triangular causal mask
    residual_logits: bool = True        # RealFormer-style logits residual
    output_attention: bool = False      # return attn weights per head
    factor: int = 5                     # ProbSparse 전용 (필요시)
    lsa: bool = False                   # Local Self-Attn scale 학습 여부(옵션)

@dataclass
class HeadConfig:
    type: Literal['prediction', 'flatten', 'pretrain'] = 'flatten'
    individual: bool = False
    head_dropout: float = 0.0
    y_range: Optional[Tuple[float, float]] = None

@dataclass
class PatchTSTConfig:
    # 데이터/윈도우
    c_in: int
    target_dim: int
    patch_len: int
    stride: int
    lookback: int = 36                  # context_window
    horizon: int = 48
    padding_patch: Optional[str] = None # 'end' or None

    # 인코더(백본)
    n_layers: int = 3
    d_model: int = 128
    d_ff: int = 256
    norm: str = 'BatchNorm'
    dropout: float = 0.0
    act: str = 'gelu'
    pre_norm: bool = False
    store_attn: bool = False
    pe: str = 'zeros'
    learn_pe: bool = True
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    verbose: bool = False

    # 서브 설정
    attn: AttentionConfig = field(default_factory=AttentionConfig)
    head: HeadConfig = field(default_factory=HeadConfig)


@dataclass
class PatchTSTConfigMonthly(PatchTSTConfig):
    lookback: int = 36
    horizon: int = 48

@dataclass
class PatchTSTConfigWeekly(PatchTSTConfig):
    lookback: int = 54
    horizon: int = 27