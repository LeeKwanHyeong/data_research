# basics.py
__all__ = ['Transpose', 'LinBnDrop', 'SigmoidRange', 'sigmoid_range', 'get_activation_fn']

from typing import Optional, Union, Callable
import torch
from torch import nn


class Transpose(nn.Module):
    """
    dims 길이가 2면 torch.Tensor.transpose 사용, 그 외엔 permute 사용.
    contiguous=True면 결과를 .contiguous()로 반환.
    """
    def __init__(self, *dims: int, contiguous: bool = False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.dims) == 2:
            y = x.transpose(self.dims[0], self.dims[1])
        else:
            y = x.permute(*self.dims)
        return y.contiguous() if self.contiguous else y


class SigmoidRange(nn.Module):
    def __init__(self, low: float, high: float):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class LinBnDrop(nn.Sequential):
    """
    BatchNorm1d, Dropout, Linear를 묶어주는 유틸.
    lin_first=True이면 Linear(+act) -> (BN) -> (Dropout)
    lin_first=False이면 (BN) -> (Dropout) -> Linear(+act)
    """
    def __init__(
        self,
        n_in: int,
        n_out: int,
        bn: bool = True,
        p: float = 0.0,
        act: Optional[Union[str, nn.Module, Callable[[], nn.Module], type]] = None,
        lin_first: bool = False,
    ):
        layers: list[nn.Module] = []

        # BN은 Linear 앞/뒤 배치에 따라 채널 수가 달라짐
        if bn:
            bn_features = n_out if lin_first else n_in
            layers.append(nn.BatchNorm1d(bn_features))

        if p and p > 0.0:
            layers.append(nn.Dropout(p))

        lin_and_act: list[nn.Module] = [nn.Linear(n_in, n_out, bias=not bn)]
        act_layer = get_activation_fn(act) if act is not None else None
        if act_layer is not None:
            lin_and_act.append(act_layer)

        # 순서 결정
        layers = (lin_and_act + layers) if lin_first else (layers + lin_and_act)
        super().__init__(*layers)


def sigmoid_range(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


def get_activation_fn(
    activation: Optional[Union[str, nn.Module, Callable[[], nn.Module], type]]
) -> Optional[nn.Module]:
    """
    - nn.Module 인스턴스: 그대로 반환
    - 문자열: 'relu' | 'gelu'
    - 콜러블/클래스: 인스턴스화해서 반환(인자 없는 생성자 가정)
    - None: None 반환
    """
    if activation is None:
        return None

    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, str):
        name = activation.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        # 필요시 확장: 'tanh', 'silu' 등
        raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or provide a callable/nn.Module')

    if callable(activation):
        # 클래스/팩토리라면 인스턴스 생성, 일반 함수라면 nn.Module이 아니므로 사용처에서 주의
        try:
            created = activation()
            if isinstance(created, nn.Module):
                return created
            raise TypeError("Callable did not return an nn.Module instance.")
        except TypeError as e:
            raise TypeError("Activation callable must be a zero-arg factory (e.g., nn.ReLU).") from e

    raise TypeError("activation must be a str, nn.Module, callable factory, or None.")