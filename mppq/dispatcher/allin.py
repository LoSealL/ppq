from typing import Collection, Dict

from mppq.dispatcher.base import DISPATCHER_TABLE, GraphDispatcher
from mppq.ir.base.graph import BaseGraph
from mppq.quant import TargetPrecision


@DISPATCHER_TABLE.register("allin")
class AllinDispatcher(GraphDispatcher):
    """Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.
    ATTENTION: this dispatcher will enable all ops in quant_types to quant_precision.
    """

    def __init__(self, graph: BaseGraph) -> None:
        self.graph = graph

    def dispatch(
        self, quant_types: Collection[str], quant_precision: TargetPrecision, **kwargs
    ) -> Dict[str, TargetPrecision]:
        """We assume all ops in origin model can be quant. This is suitable for some
        npu platform.

        Args:
            graph (BaseGraph): graph object which going to be dispatched by this
                dispatcher.
            quant_types(Set[str]): all quantable types for given platforms.

        Returns:
            Dict[str, TargetPlatform]: [description]
        """
        graph = self.graph

        dispatching_table: Dict[str, TargetPrecision] = {}
        for op in graph.operations.values():
            if op.type in quant_types:
                dispatching_table[op.name] = quant_precision
            else:
                dispatching_table[op.name] = TargetPrecision.FP32

        return dispatching_table
