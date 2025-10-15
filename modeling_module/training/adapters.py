from typing import Protocol, Any, Optional
import torch
import torch.nn as nn
PREFERRED_KEYS = ("pred", "yhat", "output", "logits")
class ModelAdapter(Protocol):
    '''
    모델별로 서로 다른 Inference/Normalization/Test Time Adaptation(TTA) 전략을 통일된 인터페이스로
    다루기 위한 Porotocol.
    - forward(model, x_batch): 배치 입력을 받아 모델의 forward 수행.
      (모델별 입력 형태가 달라도 어댑터에서 맞춤 처리)
    - reg_loss(model): 모델 구조 내부에서 발생시키는 추가 정규화 손실
      EX) Patch 간 Smoothing, Weight Penalty 등 반환
    - uses_tta(): TTA 사용 여부
    - tta_reset(model): TTA State(Memory/Context 등) 초기화. 일반적으로 Validation/Inference 시작 시 호출
    - tta_adapt(model, x_val, y_val, steps): Validation Set을 활용해 TTA 적응 수행.
      적응 과정의 Scalar Loss 등 반환할 수 있음
    '''

    def forward(
            self,
            model: nn.Module,
            x_batch: Any,
            *,
            future_exo: Optional[torch.Tensor] = None,
            mode: Optional[str] = None,
    ) -> torch.Tensor: ...

    def reg_loss(self, model: nn.Module) -> Optional[torch.Tensor]: ...

    def uses_tta(self) -> bool: ...

    def tta_reset(self, model: nn.Module): ...

    def tta_adapt(self, model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, steps: int) -> Optional[float]: ...

class DefaultAdapter:
    '''
    Default Adapter:
    - 입력이 (tuple/list)면 unpacking -> model(*x_batch) 호출, 아니면 model(x_batch) 호출
    - 별도의 정규화 손실이나 TTA를 사용하지 않는 가장 단순한 형태.
    '''

    def _call_model(self, model, x, *, future_exo=None, mode=None):
        """
                모델 시그니처가 제각각이라 안전하게 호출을 시도한다.
                - 주어진 인자가 None이면 그 조합은 건너뜀
                - 시도 순서: (x, future_exo, mode) -> (x, future_exo) -> (x, mode) -> (x)
                """
        # 1) (x, future_exo, mode)
        if future_exo is not None and mode is not None:
            try:
                return model(x, future_exo=future_exo, mode=mode)
            except TypeError as e:
                # 모델이 mode를 지원 안 할 수 있음
                # print(f"[Adapter] call (x, future_exo, mode) failed: {e}")
                pass

        # 2) (x, future_exo)
        if future_exo is not None:
            try:
                return model(x, future_exo=future_exo)
            except TypeError as e:
                # 모델이 future_exo를 지원 안 할 수 있음
                print(f"[Adapter] call (x, future_exo) failed: {e}")
                pass

        # 3) (x, mode)
        if mode is not None:
            try:
                return model(x, mode=mode)
            except TypeError as e:
                print(f"[Adapter] call (x, mode) failed: {e}")
                pass

        # 4) (x)
        return model(x)

    def _as_tensor(self, out):
        if isinstance(out, (tuple, list)):
            for item in out:
                if torch.is_tensor(item): return item
            raise TypeError(f"Model returned tuple/list without a Tensor: {type(out)}")
        if isinstance(out, dict):
            for k in PREFERRED_KEYS:
                v = out.get(k, None)
                if torch.is_tensor(v): return v
            for v in out.values():
                if torch.is_tensor(v): return v
            raise TypeError(f"Model returned dict without a Tensor value: keys={list(out.keys())}")
        if torch.is_tensor(out): return out
        raise TypeError(f"Model output is not a Tensor/tuple/dict: {type(out)}")

    def forward(self, model, x_batch, *, future_exo=None, mode=None):
        # x_batch가 dict/tuple일 수도 있으면 여기서 분기
        if isinstance(x_batch, dict):
            # dict 입력을 그대로 넘기되, exo/mode는 위의 안전 호출 경로 사용
            # -> dict 사용 모델이라면 보통 여기서 끝나므려 (x, ...)경로는 안탐

            try:
                return model(**x_batch)
            except TypeError as e:
                print(f"[Adapter] call (x, future_exo) failed: {e}")
                # dict 방식이 아니면 안전 호출로 재시도
                x_batch = x_batch.get('x', x_batch)

        if isinstance(x_batch, (tuple, list)):
            try:
                return model(*x_batch)
            except TypeError as e:
                print(f"[Adapter] call (x, future_exo) failed: {e}")
                # tuple이 (x, exo) 구조일 수 있으니 자동 분해 시도
                if len(x_batch) == 2 and future_exo is None:
                    x_only, exo = x_batch
                    return self._call_model(model, x_only, future_exo = exo, mode = mode)
                # 마지막 fallback로 첫 번째만 x로 간주
                x_batch = x_batch[0]

        out = self._call_model(model, x_batch, future_exo = future_exo, mode = mode)
        return self._as_tensor(out)


    def reg_loss(self, model): return None
    def uses_tta(self): return False
    def tta_reset(self, model): pass
    def tta_adapt(self, model, x_val, y_val, steps): return None

class PatchMixerAdapter(DefaultAdapter):
    '''
    PatchMixer 계열 모델을 위한 Adapter
    - backbone 안쪽의 'patcher' 모듈이 학습 중 계산한 정규화 손실(Ex: Patch 간 스무딩, 구조적 패널티)을
      노출한다면, 그 값을 읽어와 외부 학습 루프에서 메인 손실에 가산할 수 있도록 반환.
    '''
    def reg_loss(self, model):
        try:
            # model.backbone.patcher.last_reg_loss() 형태로 커스텀 정규화 손실을 노출한다고 가정
            patcher = getattr(getattr(model, 'backbone', None), 'patcher', None)
            if patcher is not None and hasattr(patcher, 'last_reg_loss'):
                return patcher.last_reg_loss()
        except Exception:
            # 정규화 손실 취득 실패 시 학습은 계속되도록 None 반환
            pass
        return None

class TitanAdapter(DefaultAdapter):
    '''
    Titan 계열 모델을 위한 어뎁터:
    - TTA을 위해 외부에서 전달된 팩토리(self._factory)를 통해 TTA 매니저를 생성/보관(self._tta)하고,
      검증 배치를 이용해 컨텍스트 업데이트(add_context) 및 적응(adapt)을 수행.

    Caoution:
    - uses_tta()는 Factory 유무만 판단. 실제로는 tta_reset()에서 self._tta를 생성해두는 패턴이 일반적.
    - 현재 구현의 tta_adapt는 float()을 반환하므로, Protocol Signature(Optional[torch.Tensor])와
      미묘한 타입 차이있음.
      외부 루프에서 텐서를 기대한다면 torch.tensor()로 감싸는 형태로 맞추는 것을 고려.
    '''
    def __init__(self, tta_manager_factory = None):
        # _factory: 호출 시 모델을 받아 TTA 매니저 인스턴스를 만들어주는 Callable
        self._tta = None
        self._factory = tta_manager_factory

    def uses_tta(self):
        # Factory가 제공되면 TTA 사용 대상으로 간주
        return self._factory is not None

    def tta_reset(self, model):
        '''
        TTA 상태 초기화
        - 일반적으로 Epoch 시작/ Validation 시작 시점에 호출하여 TTA 매니저를 새로 생성.
        - self._factory가 없다면 TTA를 비활성화 상태(None)로 둔다.
        '''
        # Factory가 주어진 경우에만 TTA 매니저 생성
        # (Optional) self._tta = self._factory(model)
        self._tta = None

    def tta_adapt(self, model, x_val, y_val, steps):
        """
        검증 배치를 사용한 테스트 타임 적응 절차
        1) Context Update(self._tta.add_context)
        2) Adaptation (self._tta.adapt) 및 Scalar loss 반환 (Casting float)

        return:
        - 적응 과정의 Scalar(ex: loss)를 float으로 반환하거나, TTA 미사용 시 None
        """
        if not self._tta: return None
        # add context & adapt
        self._tta.add_context(x_val)
        return float(self._tta.adapt(x_val, y_val, steps = steps))
