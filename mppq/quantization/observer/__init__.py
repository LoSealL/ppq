from typing import Dict, List, Optional, Sequence

import torch

from mppq.executor import QuantOPRuntimeHook
from mppq.ir.base.opdef import Variable
from mppq.ir.base.quantize import QuantableOperation
from mppq.logger import error, info
from mppq.quant import QuantizationStates, TensorQuantizationConfig

from .base import OBSERVER_TABLE, BaseTensorObserver
from .dbdc import TorchDbdcObserver
from .floating import ConstantObserver, DirectMSEObserver
from .hist import TorchHistObserver, TorchHistogramObserver
from .isotone import TorchIsotoneObserver
from .min_max import TorchMinMaxObserver
from .mse import TorchMSEObserver
from .percentile import TorchPercentileObserver

__all__ = [
    "TorchDbdcObserver",
    "ConstantObserver",
    "DirectMSEObserver",
    "TorchHistObserver",
    "TorchHistogramObserver",
    "TorchIsotoneObserver",
    "TorchMinMaxObserver",
    "TorchMSEObserver",
    "TorchPercentileObserver",
]


def build_observer(
    variable: Variable, config: TensorQuantizationConfig
) -> BaseTensorObserver:
    algorithm = str(config.observer_algorithm.lower())
    if algorithm not in OBSERVER_TABLE:
        error(f"observer algorithm {algorithm} not found")
        info(f"{OBSERVER_TABLE}")
        raise KeyError
    return OBSERVER_TABLE[algorithm](watch_on=variable, quant_cfg=config)


class CalibrationHook(QuantOPRuntimeHook):
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

    def __str__(self) -> str:
        return "".join(
            [observer.__str__() + "\n" for _, observer in self._observer_table.items()]
        )


class OperationObserver:
    def __init__(
        self,
        operation: QuantableOperation,
        monitor_parameter: bool = True,
        monitor_outputs: bool = True,
        monitor_inputs: bool = True,
    ) -> None:
        if not isinstance(operation, QuantableOperation):
            raise TypeError(
                f"Only QuantableOP instance can apply an Observer, "
                f"while {type(operation)} was given."
            )

        self._operation = operation
        self._hook = self.build_hook(
            monitor_parameter=monitor_parameter,
            monitor_outputs=monitor_outputs,
            monitor_inputs=monitor_inputs,
        )

    def render_quantization_config(self):
        self.hook.render_quantization_config()

    def build_hook(
        self, monitor_parameter: bool, monitor_outputs: bool, monitor_inputs: bool
    ) -> CalibrationHook:
        assert isinstance(self._operation, QuantableOperation)
        observer_table = {}
        for var, config in zip(
            self._operation.inputs, self._operation.config.input_quantization_config
        ):
            if config.state == QuantizationStates.INITIAL:
                if var in self._operation.parameters and monitor_parameter:
                    observer_table[config] = build_observer(var, config)
                elif monitor_inputs:
                    observer_table[config] = build_observer(var, config)

        if monitor_outputs:
            for var, config in zip(
                self._operation.outputs,
                self._operation.config.output_quantization_config,
            ):
                if config.state == QuantizationStates.INITIAL:
                    observer_table[config] = build_observer(var, config)

        return CalibrationHook(operation=self._operation, observer_table=observer_table)

    @property
    def hook(self) -> CalibrationHook:
        return self._hook

    def report(self) -> str:
        return str(self._hook)
