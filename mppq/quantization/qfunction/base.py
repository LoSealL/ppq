from typing import Protocol

from torch import Tensor

from mppq.quant import TensorQuantizationConfig


class BaseQuantFunction(Protocol):
    def __call__(self, tensor: Tensor, config: TensorQuantizationConfig) -> Tensor:
        return tensor
