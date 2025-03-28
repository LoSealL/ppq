from abc import ABCMeta, abstractmethod
from typing import Any, Type

from mppq.ir.base.opdef import Variable
from mppq.quant import QuantizationStates, TensorQuantizationConfig
from mppq.register import Registry


class BaseTensorObserver(metaclass=ABCMeta):
    """A base class for all tensor observers."""

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        self._watch_on = watch_on
        self._quant_cfg = quant_cfg

    @abstractmethod
    def observe(self, value: Any):
        raise NotImplementedError("Implement this function first.")

    @abstractmethod
    def render_quantization_config(self):
        raise NotImplementedError("Implement this function first.")

    def __repr__(self) -> str:
        return (
            "PPQ Tensor Observer ("
            + self.__class__.__name__
            + ") mount on variable "
            + self._watch_on.name
            + " observing algorithm: "
            + self._quant_cfg.observer_algorithm
        )

    def report(self) -> str:
        if self._quant_cfg.state == QuantizationStates.ACTIVATED:
            return (
                f"Observer on Variable {self._watch_on.name}, "
                f"computed scale: {self._quant_cfg.scale}, "
                f"computed offset: {self._quant_cfg.offset}\n"
            )
        return ""


OBSERVER_TABLE: Registry[Type[BaseTensorObserver]] = Registry("OBSERVER_TABLE")
