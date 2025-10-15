# resources/dto.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple
import torch

class SupportsStateDict(Protocol):
    def state_dict(self) -> dict: ...

class TrainerLike(Protocol):
    def fit(self) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        Returns:
          trained_model: Any
          train_metrics: Dict[str, Any]
          val_metrics: Dict[str, Any]
        """
        ...

@dataclass(frozen=True)
class TrainingRequest:
    trainer: TrainerLike

@dataclass(frozen=True)
class ForecastRequest:
    models: Dict[str, torch.nn.Module]
    data_loader: Optional[Iterable] = None  # 외부에서 세팅한 로더 주입 (None이면 서비스 보유 로더 사용)
    device: str = "cpu"
    future_exo_cb: Optional[Callable[[int, int], torch.Tensor]] = None
    horizon: int = 120
    target_channel: int = 0
