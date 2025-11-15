# configs.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TitanConfig:
    lookback: int = 52
    horizon: int = 27
    d_model: int = 256
    input_dim: int = 1
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # LMM(메모리)
    use_lmm: bool = False
    contextual_mem_size: int = 256
    persistent_mem_size: int = 64

    # RevIN
    use_revin: bool = True
    revin_use_std: bool = False
    revin_subtract_last: bool = True
    revin_affine: bool = True

    # Expander 옵션(계절/저주파 강화)
    use_temporal_expander: bool = True
    expander_f_out: int = 32
    expander_max_harmonics: int = 6
    expander_n_harmonics: int = 6
    expander_use_conv: bool = True

    # Exogenous
    use_exogenous: bool = True
    exo_dim: int = 2

    # 출력 제약
    final_clamp_nonneg: bool = True


    # Head 유형: 'expander' | 'linear' | 'seq2seq'
    head_type: str = 'expander'



    # Seq2Seq 디코더 옵션
    dec_n_layers: int = 2
    dec_n_heads: int = 4
    dec_d_ff: int = 512
    dec_dropout: float = 0.1
