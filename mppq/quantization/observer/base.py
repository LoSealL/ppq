from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Sequence, Type

import torch

from mppq.executor.base import QuantRuntimeHook
from mppq.ir.base.opdef import Operation, Variable
from mppq.ir.base.quantize import QuantableOperation
from mppq.logger import error, info
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


class CalibrationHook(QuantRuntimeHook):
    """A runtime hook for operation observer."""

    def __init__(
        self,
        operation: QuantableOperation,
        observer_table: Dict[Variable, BaseTensorObserver],
    ) -> None:
        self._operation = operation
        self._observer_table = observer_table
        super().__init__(operation)

    def pre_forward_hook(
        self,
        inputs: Sequence[torch.Tensor],
        quant_inputs: Sequence[torch.Tensor] = (),
        quant_configs: Sequence[TensorQuantizationConfig] = (),
        **kwargs,
    ):
        for input_var, quant_config in zip(inputs, quant_configs):
            if quant_config in self._observer_table:
                observer = self._observer_table[quant_config]
                observer.observe(input_var)
        return super().pre_forward_hook(inputs, quant_inputs, quant_configs)

    def post_forward_hook(
        self,
        outputs: Sequence[torch.Tensor],
        quant_outputs: Sequence[torch.Tensor] = (),
        quant_configs: Sequence[TensorQuantizationConfig] = (),
        **kwargs,
    ):
        for output_var, quant_config in zip(outputs, quant_configs):
            if quant_config in self._observer_table:
                observer = self._observer_table[quant_config]
                observer.observe(output_var)
        return super().post_forward_hook(outputs, quant_outputs, quant_configs)

    def render_quantization_config(self):
        for _, observer in self._observer_table.items():
            observer.render_quantization_config()
            observer.report()

    def __repr__(self) -> str:
        return "".join(
            [observer.__str__() + "\n" for _, observer in self._observer_table.items()]
        )


class OperationObserver:
    def __init__(
        self,
        operation: Operation,
        monitor_parameter: bool = True,
        monitor_outputs: bool = True,
        monitor_inputs: bool = True,
    ) -> None:
        self._operation = operation
        self._hook = OperationObserver.build_hook(
            operation=operation,
            monitor_parameter=monitor_parameter,
            monitor_outputs=monitor_outputs,
            monitor_inputs=monitor_inputs,
        )

    @staticmethod
    def build_observer(
        variable: Variable, config: TensorQuantizationConfig
    ) -> BaseTensorObserver:
        algorithm = str(config.observer_algorithm.lower())
        if algorithm not in OBSERVER_TABLE:
            error(f"observer algorithm {algorithm} not found")
            info(f"{OBSERVER_TABLE}")
            raise KeyError
        return OBSERVER_TABLE[algorithm](watch_on=variable, quant_cfg=config)

    @staticmethod
    def build_hook(
        operation: Operation,
        monitor_parameter: bool,
        monitor_outputs: bool,
        monitor_inputs: bool,
    ) -> CalibrationHook:
        if not isinstance(operation, QuantableOperation):
            raise TypeError(
                f"Only QuantableOP instance can apply an Observer, "
                f"while {type(operation)} was given."
            )
        observer_table = {}
        for var, cfg in zip(
            operation.inputs, operation.config.input_quantization_config
        ):
            if cfg.state == QuantizationStates.INITIAL:
                if var.is_parameter and monitor_parameter:
                    observer_table[cfg] = OperationObserver.build_observer(var, cfg)
                elif not var.is_parameter and monitor_inputs:
                    observer_table[cfg] = OperationObserver.build_observer(var, cfg)

        if monitor_outputs:
            for var, cfg in zip(
                operation.outputs,
                operation.config.output_quantization_config,
            ):
                if cfg.state == QuantizationStates.INITIAL:
                    observer_table[cfg] = OperationObserver.build_observer(var, cfg)

        return CalibrationHook(operation=operation, observer_table=observer_table)

    def render_quantization_config(self):
        self._hook.render_quantization_config()

    @property
    def hook(self) -> CalibrationHook:
        return self._hook

    def report(self) -> str:
        """print debug messages"""
        return str(self._hook)
