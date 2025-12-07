from __future__ import annotations
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "TitanBaseModel",
    "TitanLMMModel",
    "TitanSeq2SeqModel",
]

from modeling_module.models.common_layers.RevIN import RevIN

# --- 패키지/로컬 모두 대응 가능한 유연한 임포트 ---
try:
    from modeling_module.models.Titan.backbone import MemoryEncoder
    from modeling_module.models.Titan.common.decoder import TitanDecoder
except Exception:
    from backbone import MemoryEncoder          # type: ignore
    from decoder import TitanDecoder            # type: ignore

# TitanConfig 는 configs.py 에 정의되어 있다고 가정
try:
    from modeling_module.models.Titan.common.configs import TitanConfig
except Exception:
    try:
        from configs import TitanConfig         # type: ignore
    except Exception:
        TitanConfig = None                      # type: ignore


def _merge_cfg_kwargs(cfg_obj, **kwargs):
    """cfg(dataclass)와 kwargs 병합. kwargs 우선."""
    cfg_dict = asdict(cfg_obj) if cfg_obj is not None else {}
    cfg_dict.update(kwargs)
    return cfg_dict


class _TitanBase(nn.Module):
    """
    공통 베이스: config 보관 + Encoder/Decoder 조립
    - 외생변수/비음수 클램프 등 공통 옵션은 여기서 통일
    """
    def __init__(self, *, config: Optional["TitanConfig"]=None, **kwargs):
        super().__init__()
        params = _merge_cfg_kwargs(config, **kwargs)

        # Core
        self.input_dim: int = int(params["input_dim"])
        self.lookback: int = int(params["lookback"])
        self.horizon: int = int(params["horizon"])
        self.d_model: int = int(params.get("d_model", 256))
        self.n_layers: int = int(params.get("n_layers", 3))
        self.n_heads: int = int(params.get("n_heads", 4))
        self.d_ff: int = int(params.get("d_ff", 512))
        self.dropout: float = float(params.get("dropout", 0.1))

        # RevIN 옵션 (configs.py에 이미 존재)
        self.use_revin = bool(params.get("use_revin", True))
        self.revin_subtract_last = bool(params.get("revin_subtract_last", False))
        self.revin_affine = bool(params.get("revin_affine", True))
        self.revin_use_std = bool(params.get("revin_use_std", True))

        # Memory/LMM
        self.contextual_mem_size: int = int(params.get("contextual_mem_size", 256))
        self.persistent_mem_size: int = int(params.get("persistent_mem_size", 64))

        # Exogenous
        self.use_exogenous: bool = bool(params.get("use_exogenous", False))
        self.exo_dim: int = int(params.get("exo_dim", 0))
        self.use_calendar_exo: bool = bool(params.get("use_calendar_exo", False))

        # Output constraint
        self.final_clamp_nonneg: bool = bool(params.get("final_clamp_nonneg", True))

        # Seq2Seq decoder 전용 파라미터(필요 시 사용)
        self.dec_layers: int = int(params.get("dec_layers", 2))
        self.dec_heads: int = int(params.get("dec_heads", 4))
        self.dec_d_ff: int = int(params.get("dec_d_ff", 512))
        self.dec_dropout: float = float(params.get("dec_dropout", 0.1))

        # 원본 config 보관(트레이너가 참조 가능)
        self.config = config

        self.target_channel = int(params.get("target_channel", 0))
        self.revin = RevIN(
            num_features=1,  # 단일 타깃 채널 기준
            affine=self.revin_affine,
            subtract_last=self.revin_subtract_last,
            use_std=self.revin_use_std
        ) if self.use_revin else None

        # Encoder
        self.encoder = MemoryEncoder(
            self.input_dim,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.d_ff,
            self.contextual_mem_size,
            self.persistent_mem_size,
            self.dropout,
            use_context_update=False  # 일단 안전하게 False로 고정하거나 config 옵션 연동
        )

    @classmethod
    def from_config(cls, config: "TitanConfig"):
        return cls(config=config)

    def _clamp(self, y: torch.Tensor) -> torch.Tensor:
        return y.clamp_min(0) if self.final_clamp_nonneg else y

    # 입력 x: [B, L, C] -> RevIN(norm) -> [B, L, C]
    def _maybe_revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        if (self.revin is None) or (x.size(-1) == 0):
            print('revin is None So cannot normalize')
            return x
        # 타깃 채널만 정규화/복원(멀티채널 안전)
        tc = self.target_channel
        x_t = x[:, :, tc:tc+1]
        x_t = self.revin(x_t, mode='norm')   # [B,L,1]
        x = x.clone()
        x[:, :, tc:tc+1] = x_t
        return x

    # 출력 y: [B, H] -> RevIN(denorm) -> [B, H]
    def _maybe_revin_denorm(self, y: torch.Tensor) -> torch.Tensor:
        if (self.revin is None) or (y.dim() != 2):
            print('revin is None so cannot denormalize')
            return y
        # RevIN은 [B,L,C] 형태를 기대하므로 H축을 L로 보고, C=1로 맞춰 복원
        y_in = y.unsqueeze(-1)       # [B, H, 1]
        y_out = self.revin(y_in, mode='denorm')  # [B, H, 1]
        return y_out.squeeze(-1)

    def _clamp(self, y: torch.Tensor) -> torch.Tensor:
        return y.clamp_min(0) if self.final_clamp_nonneg else y


class TitanBaseModel(_TitanBase):
    """
    Encoder-only: TitanDecoder를 수평(H) 차원 투영기로 사용하고 Linear로 1채널 예측.
    """
    def __init__(self, *, config: Optional["TitanConfig"]=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.decoder = TitanDecoder(
            d_model=self.d_model,
            n_layers=1,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            horizon=self.horizon,
            exo_dim=(self.exo_dim if self.use_exogenous else 0),
        )
        self.proj = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor, *, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        # 1) RevIN norm (입력 전처리) ---------------------
        x = self._maybe_revin_norm(x)  # [B,L,C]

        # 2) Encoder-Decoder-Head -------------------------
        memory = self.encoder(x)  # [B,L,D]
        dec = self.decoder(memory, future_exo)  # [B,H,D]
        y = self.proj(dec).squeeze(-1)  # [B,H]

        # 폭주 로그 확인용
        # y_before = y.detach().clone()


        # 3) RevIN denorm (출력 복원) ---------------------
        y = self._maybe_revin_denorm(y)  # [B,H]


        # 폭주 로그 확인용
        # if (self.revin is not None) and (y.dim() == 2):
        #     with torch.no_grad():
        #         b0 = 0
        #         print(f"[TitanDBG] model={getattr(self, 'model_name', self.__class__.__name__)}")
        #         print(f"  y_before[0,:5]={y_before[b0, :5].tolist()}")
        #         print(f"  y_after [0,:5]={y[b0, :5].tolist()}")
        #         # RevIN 내부 통계가 public이라면 찍기 (구현에 맞게 조정)
        #         if hasattr(self.revin, 'mean'):
        #             m = getattr(self.revin, 'mean', None)
        #             s = getattr(self.revin, 'std', None)
        #             last = getattr(self.revin, 'last', None)
        #             if m is not None:
        #                 print(
        #                     f"  revin.mean[0, :5, 0]={m[b0, :5, 0].detach().cpu().numpy() if m.dim() == 3 else m[b0, :5].detach().cpu().numpy()}")
        #             if s is not None:
        #                 print(
        #                     f"  revin.std [0, :5, 0]={s[b0, :5, 0].detach().cpu().numpy() if s.dim() == 3 else s[b0, :5].detach().cpu().numpy()}")
        #             if last is not None:
        #                 print(f"  revin.last[0,0,0]={float(last[b0, 0, 0])}")

        # 4) 최종 제약(비음수 등) -------------------------

        return self._clamp(y)


class TitanLMMModel(_TitanBase):
    """
    LMM 특화 디코딩이 필요하면 TitanDecoder 내부에서 분기하도록 구성(여기선 공용 Decoder 사용).
    """
    def __init__(self, *, config: Optional["TitanConfig"]=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.decoder = TitanDecoder(
            d_model=self.d_model,
            n_layers=1,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            horizon=self.horizon,
            exo_dim=(self.exo_dim if self.use_exogenous else 0),
        )
        self.proj = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor, *, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        # 1) RevIN norm (입력 전처리) ---------------------
        x = self._maybe_revin_norm(x)  # [B,L,C]
        memory = self.encoder(x)
        dec = self.decoder(memory, future_exo)
        y = self.proj(dec).squeeze(-1)
        # 3) RevIN denorm (출력 복원) ---------------------
        y = self._maybe_revin_denorm(y)  # [B,H]
        return self._clamp(y)


class TitanSeq2SeqModel(_TitanBase):
    """
    Seq2Seq: 다층 디코더를 이용하여 미래 H 단계의 컨텍스트를 생성 후 1채널 예측.
    """
    def __init__(self, *, config: Optional["TitanConfig"]=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.decoder = TitanDecoder(
            d_model=self.d_model,
            n_layers=self.dec_layers,
            n_heads=self.dec_heads,
            d_ff=self.dec_d_ff,
            dropout=self.dec_dropout,
            horizon=self.horizon,
            exo_dim=(self.exo_dim if self.use_exogenous else 0),
        )
        self.proj = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor, *, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        # 1) RevIN norm (입력 전처리) ---------------------
        x = self._maybe_revin_norm(x)  # [B,L,C]
        memory = self.encoder(x)               # [B, L, D]
        dec = self.decoder(memory, future_exo) # [B, H, D]
        y = self.proj(dec).squeeze(-1)         # [B, H]
        # 3) RevIN denorm (출력 복원) ---------------------
        y = self._maybe_revin_denorm(y)  # [B,H]
        return self._clamp(y)
