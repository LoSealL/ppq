"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import (
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

import torch
from torch import Tensor

from mppq.executor.op.base import DEFAULT_BACKEND_TABLE, TorchBackendContext
from mppq.ir.base.graph import BaseGraph
from mppq.ir.base.opdef import Operation
from mppq.ir.base.quantize import QuantableOperation
from mppq.quant import BuiltinPlatform, TensorQuantizationConfig

OPERATION_FORWARD_TABLE = {
    platform.value: deepcopy(DEFAULT_BACKEND_TABLE) for platform in BuiltinPlatform
}

GraphInput = TypeVar("GraphInput", Sequence[Tensor], Mapping[str, Tensor], Tensor)


class RuntimeHook:
    """RuntimeHook is an abstract class designed for executor customizing.

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(self, operation: Operation, **kwargs) -> None:
        self._hook_to = operation

    def pre_forward_hook(self, inputs: Sequence[Tensor], **kwargs) -> List[Tensor]:
        """user-customized pre-processing procedure of input data.

        Args:
            inputs (list): a list includes all input data.

        Returns:
            list: a list includes all input data(processed).
        """
        return list(inputs)

    def post_forward_hook(self, outputs: Sequence[Tensor], **kwargs) -> List[Tensor]:
        """user-customized post-processing procedure of output data.

        Args:
            inputs (list): a list includes all output data.

        Returns:
            list: a list includes all output data(processed).
        """
        return list(outputs)


class QuantRuntimeHook(RuntimeHook):
    """QuantRuntimeHook is an abstract class designed for executor customizing."""

    def __init__(self, operation: Operation, **kwargs) -> None:
        if not isinstance(operation, QuantableOperation):
            raise TypeError(
                "You are trying to bind a QuantRuntimeHook "
                f"to a non-quantized operation {operation}."
            )
        super().__init__(operation, **kwargs)

    def pre_forward_hook(
        self,
        inputs: Sequence[Tensor],
        quant_inputs: Sequence[Tensor] = (),
        quant_configs: Sequence[TensorQuantizationConfig] = (),
        **kwargs,
    ) -> List[Tensor]:
        assert len(inputs) == len(quant_inputs) == len(quant_configs)
        return list(quant_inputs)

    def post_forward_hook(
        self,
        outputs: Sequence[Tensor],
        quant_outputs: Sequence[Tensor] = (),
        quant_configs: Sequence[TensorQuantizationConfig] = (),
        **kwargs,
    ) -> List[Tensor]:
        assert len(outputs) == len(quant_outputs) == len(quant_configs)
        return list(quant_outputs)


class BaseGraphExecutor(metaclass=ABCMeta):
    """PPQ Base Graph Executor.

    Args:
        Callable ([type]): [description]
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(
        self, graph: BaseGraph, target_platform: int = BuiltinPlatform.UNSPECIFIED
    ) -> None:
        self.load_graph(graph=graph)
        self.target_platform = target_platform

    @property
    def graph(self) -> BaseGraph:
        return self._graph

    def load_graph(self, graph: BaseGraph):
        self._graph = graph
        self._executing_order = self._graph.topological_sort()

    def _prepare_input(self, inputs: Optional[GraphInput]) -> Dict[str, Tensor]:
        inputs_dictionary = self._graph.inputs
        if len(inputs_dictionary) == 0:
            assert (
                inputs is None
            ), "Graph do not need any inputs. please set your inputs to be None."
            return {}

        if isinstance(inputs, torch.Tensor):
            assert (
                len(inputs_dictionary) == 1
            ), "Graph needs more than one input, while only one tensor was given."
            return {list(inputs_dictionary.keys())[0]: inputs}
        elif isinstance(inputs, Mapping):
            assert len(inputs_dictionary) == len(inputs), (
                f"Inputs format misunderstood. Given inputs has "
                f"{len(inputs)} elements, while graph needs {len(inputs_dictionary)}"
            )
            return {k: v for k, v in inputs.items()}
        elif isinstance(inputs, Sequence):
            assert len(inputs_dictionary) == len(inputs), (
                f"Inputs format misunderstood. Given inputs has "
                f"{len(inputs)} elements, while graph needs {len(inputs_dictionary)}"
            )
            return {k: v for k, v in zip(inputs_dictionary, inputs)}
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

    @abstractmethod
    def forward(
        self,
        inputs: GraphInput,
        output_names: Optional[List[str]] = None,
        hooks: Optional[Mapping[str, RuntimeHook]] = None,
    ) -> List[torch.Tensor]:
        """Forward a graph from given inputs to required output names.

        If output_names is not given, all graph outputs will be returned.
        hook is called before and after each operation's forward.
        """
        raise NotImplementedError

    @abstractmethod
    def tracing_operation_meta(
        self,
        inputs: GraphInput,
        output_names: Optional[List[str]] = None,
    ) -> None:
        raise NotImplementedError("Please implement this function first.")

    @overload
    def forward_single_operation(
        self,
        op: Operation,
        inputs: List[Tensor],
        ctx: Optional[TorchBackendContext] = None,
        return_list: Literal[True] = True,
    ) -> Tuple[Tensor, ...]:
        """"""

    @overload
    def forward_single_operation(
        self,
        op: Operation,
        inputs: List[Tensor],
        ctx: Optional[TorchBackendContext] = None,
        return_list: bool = True,
    ) -> Tuple[Tensor, ...] | Tensor:
        """"""

    def forward_single_operation(
        self,
        op: Operation,
        inputs: List[Tensor],
        ctx: Optional[TorchBackendContext] = None,
        return_list: bool = True,
    ) -> Tuple[Tensor, ...] | Tensor:
        """Forward a single operation with given inputs."""
        f = OPERATION_FORWARD_TABLE[self.target_platform][op.type]
        ret = f(op, inputs, ctx=ctx)
        if return_list and not isinstance(ret, (list, tuple)):
            return (ret,)
        return ret

    def __call__(
        self,
        inputs: GraphInput,
        output_names: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        return self.forward(inputs=inputs, output_names=output_names)

    def __repr__(self) -> str:
        return f"PPQ GraphExecuter Object: {self.__hash__()}"
