from typing import Protocol, Any, Optional, List, Dict
import inspect
import torch
import torch.nn as nn

PREFERRED_KEYS = ("pred", "yhat", "output", "logits")


class ModelAdapter(Protocol):
    def forward(
        self,
        model: nn.Module,
        x_batch: Any,
        *,
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,  # [B,L,E_c] float32
        past_exo_cat: Optional[torch.Tensor] = None,   # [B,L,E_k] long
        part_ids: Optional[List[str]] = None,          # 길이 B
        mode: Optional[str] = None,
    ) -> torch.Tensor: ...

    def reg_loss(self, model: nn.Module) -> Optional[torch.Tensor]: ...
    def uses_tta(self) -> bool: ...
    def tta_reset(self, model: nn.Module): ...
    def tta_adapt(self, model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, steps: int) -> Optional[float]: ...


def _model_accepts_kw(model: nn.Module, kw: str) -> bool:
    try:
        sig = inspect.signature(model.forward)
        return kw in sig.parameters
    except Exception:
        return False


class DefaultAdapter:
    """
    - 모델 forward 시그니처를 점검하여 지원하는 키워드만 안전하게 전달
    - future_exo/past_exo_cont/past_exo_cat/part_ids/mode 모두 '선별 전달'
    - RevIN denorm 등은 필요 시 개별 모델/어댑터에서 처리
    """

    # -------------------- utils --------------------
    def _call_model(
        self,
        model: nn.Module,
        x: Any,
        *,
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ):
        accepts: Dict[str, bool] = {
            "future_exo": _model_accepts_kw(model, "future_exo"),
            "past_exo_cont": _model_accepts_kw(model, "past_exo_cont"),
            "past_exo_cat": _model_accepts_kw(model, "past_exo_cat"),
            "part_ids": _model_accepts_kw(model, "part_ids"),
            "mode": _model_accepts_kw(model, "mode"),
        }

        kwargs = {}
        if future_exo is not None and accepts["future_exo"]:
            kwargs["future_exo"] = future_exo
        if past_exo_cont is not None and accepts["past_exo_cont"]:
            kwargs["past_exo_cont"] = past_exo_cont
        if past_exo_cat is not None and accepts["past_exo_cat"]:
            kwargs["past_exo_cat"] = past_exo_cat
        if part_ids is not None and accepts["part_ids"]:
            kwargs["part_ids"] = part_ids
        if mode is not None and accepts["mode"]:
            kwargs["mode"] = mode

        return model(x, **kwargs)

    def _as_tensor(self, out):
        if isinstance(out, (tuple, list)):
            for item in out:
                if torch.is_tensor(item):
                    return item
            raise TypeError(f"Model returned tuple/list without a Tensor: {type(out)}")
        if isinstance(out, dict):
            for k in PREFERRED_KEYS:
                v = out.get(k, None)
                if torch.is_tensor(v):
                    return v
            for v in out.values():
                if torch.is_tensor(v):
                    return v
            raise TypeError(f"Model returned dict without a Tensor value: keys={list(out.keys())}")
        if torch.is_tensor(out):
            return out
        raise TypeError(f"Model output is not a Tensor/tuple/dict: {type(out)}")

    # -------------------- main --------------------
    def forward(
        self,
        model: nn.Module,
        x_batch: Any,
        *,
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:

        # dict 입력: 우선 **kwargs로 시도
        if isinstance(x_batch, dict):
            try:
                return self._as_tensor(model(**x_batch))
            except TypeError:
                x_batch = x_batch.get("x", x_batch)

        # tuple/list 입력: 바로 언패킹 → 실패 시 (x, exo) 패턴만 보조 처리
        if isinstance(x_batch, (tuple, list)):
            try:
                return self._as_tensor(model(*x_batch))
            except TypeError:
                if len(x_batch) == 2 and future_exo is None:
                    x_only, exo = x_batch
                    out = self._call_model(
                        model, x_only,
                        future_exo=exo,
                        past_exo_cont=past_exo_cont,
                        past_exo_cat=past_exo_cat,
                        part_ids=part_ids,
                        mode=mode,
                    )
                    return self._as_tensor(out)
                x_batch = x_batch[0]

        out = self._call_model(
            model, x_batch,
            future_exo=future_exo,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
            part_ids=part_ids,
            mode=mode,
        )
        return self._as_tensor(out)

    # -------------------- optional hooks --------------------
    def reg_loss(self, model): return None
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
    def __init__(self, tta_manager_factory=None):
        self._tta = None
        self._factory = tta_manager_factory

    def uses_tta(self):
        return self._factory is not None

    def tta_reset(self, model):
        self._tta = None

    def tta_adapt(self, model, x_val, y_val, steps):
        if not self._tta:
            return None
        self._tta.add_context(x_val)
        return float(self._tta.adapt(x_val, y_val, steps=steps))