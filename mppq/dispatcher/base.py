from abc import ABCMeta, abstractmethod
from typing import Collection, Dict, Set, Type

from mppq.common import INT_CALCULATION_OPS, SOI_DATA_GENERATOR_OPS
from mppq.ir.base.graph import BaseGraph
from mppq.ir.base.opdef import Operation
from mppq.quant import TargetPrecision
from mppq.register import Registry


class GraphDispatcher(metaclass=ABCMeta):
    r"""Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.

    For the most part, all operations within graph can be partitioned into quantable
    operations, Shape-Or-Index (SOI) related operations and remaining operations, all
    sub classes of GraphDispatcher will give an implementation of function "dispatch"
    to send all operations to their proper platform.

    ATTENTION:

        platform attribute will greatly affect quantizer's quantization logic, and the
        execution result.
        If operation is sent to a quantable platform, then its inputs and outputs will
        be quantized if necessary.
        if operation is classified as shape-or-index related operation, then its
        execution will be taken with cpu.
        if operation is sent to a fp32 platform, then its inputs and outputs shall
        never be quantized.
    """

    @abstractmethod
    def __init__(self, graph: BaseGraph, **kwargs) -> None:
        raise NotImplementedError("Impl this first.")

    @abstractmethod
    def dispatch(
        self, quant_types: Collection[str], **kwargs
    ) -> Dict[str, TargetPrecision]:
        """Graph Dispatcher splits a graph into parts, each part of graph will
        be sent to a specific platform for further execution and quantization.
        """
        raise NotImplementedError("Impl this first.")

    def __call__(
        self, quant_types: Collection[str], **kwargs
    ) -> Dict[str, TargetPrecision]:
        return self.dispatch(quant_types, **kwargs)


def value_tracing_pattern(x: Operation, y: Operation) -> bool:
    if y.type in INT_CALCULATION_OPS:
        # shape can go through above operations as a input, under this circumstance,
        # their output should still be a tensor of shape.
        # However if shape was feed as a parameter for those operations, then their
        # outputs are irrelevant with shape flow.
        return y.inputs[0].source_op == x
    if y.type == "ScatterND":
        # ScatterND has 2 quant input.
        return y.inputs[0].source_op == x or y.inputs[-1].source_op == x

    if y.type in {"ConstantOfShape", "Shape", "NonMaxSuppression"}:
        # Inputs: (1)
        #   input : T
        #     1D tensor. The shape of the expected output tensor.
        #     If empty tensor is given, the output would be a scalar.
        #     All values must be >= 0.
        # see also:
        # https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConstantOfShape
        return False

    return True


def reverse_tracing_pattern(x: Operation, y: Operation) -> bool:
    if y.type in ("Shape", "TopK"):
        return False
    if x.type in INT_CALCULATION_OPS:
        return y == x.inputs[0].source_op
    if x.type == "ScatterND":
        return y == x.inputs[0].source_op or y == x.inputs[-1].source_op
    if x.type in {"NonMaxSuppression", "Shape"}:
        # remove constant of shape from here can speed up.
        return False
    return True


def soi_receivers(graph: BaseGraph) -> Set[Operation]:
    r"""Get operations in the graph that one of inputs is SOI."""
    receivers = set()
    for operation in graph.operations.values():
        for idx, plat in enumerate(operation.socket.in_plat):
            if plat in (TargetPrecision.SOI, TargetPrecision.FP32):
                receivers.add(operation.inputs[idx].source_op)
    receivers.discard(None)
    return receivers


def soi_generators(graph: BaseGraph) -> Set[Operation]:
    r"""Get operations in the graph that one of outputs may be SOI."""
    return {i for i in graph.operations.values() if i.type in SOI_DATA_GENERATOR_OPS}


DISPATCHER_TABLE: Registry[Type[GraphDispatcher]] = Registry("DISPATCHER_TABLE")
