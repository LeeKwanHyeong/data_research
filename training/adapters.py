from typing import Protocol, Any, Optional
import torch
import torch.nn as nn

class ModelAdapter(Protocol):
    def forward(self, model: nn.Module, x_batch: Any) -> torch.Tensor: ...
    def reg_loss(self, model: nn.Module) -> Optional[torch.Tensor]: ...
    def uses_tta(self) -> bool: ...
    def tta_reset(self, model: nn.Module): ...
    def tta_adapt(self, model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, steps: int) -> Optional[torch.Tensor]: ...

class DefaultAdapter:
    def forward(self, model, x_batch):
        if isinstance(x_batch, (tuple, list)):
            return model(*x_batch)
        return model(x_batch)

    def reg_loss(self, model):
        return None

    def uses_tta(self): return False
    def tta_reset(self, model): pass
    def tta_adapt(self, model, x_val, y_val, steps): return None

class PatchMixerAdapter(DefaultAdapter):
    def reg_loss(self, model):
        try:
            patcher = getattr(getattr(model, 'backbone', None), 'patcher', None)
            if patcher is not None and hasattr(patcher, 'last_reg_loss'):
                return patcher.last_reg_loss()
        except Exception:
            pass
        return None

class TitanAdapter(DefaultAdapter):
    def __init__(self, tta_manager_factory = None):
        self._tta = None
        self._factory = tta_manager_factory

    def uses_tta(self): return self._factory is not None

    def tta_adapt(self, model, x_val, y_val, steps):
        if not self._tta: return None
        # add context & adapt
        self._tta.add_context(x_val)
        return float(self._tta.adapt(x_val, y_val, steps = steps))
