"""
This File Contains Legacy Dispatchers.
Refer to ppq.scheduler.perseus for updated implementation.
"""

from typing import Collection, Dict

from mppq.ir.base.graph import BaseGraph
from mppq.ir.search import SearchableGraph
from mppq.quant import TargetPrecision

from .base import (
    DISPATCHER_TABLE,
    GraphDispatcher,
    reverse_tracing_pattern,
    soi_generators,
    soi_receivers,
    value_tracing_pattern,
)


@DISPATCHER_TABLE.register("conservative")
class ConservativeDispatcher(GraphDispatcher):
    r"""Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.

    For the most part, all operations within graph can be partitioned into quantable
    operations, Shape-Or-Index related operations and remaining operations, all sub
    classes of GraphDispatcher will give an implementation of function "dispatch" to
    send all operations to their proper platform.

    Conservative Dispatcher cuts graph in a conservative way, which means it takes as
    much as possible operations into fp32 platform.

    ATTENTION:

        platform attribute will greatly affect quantizer's quantization logic, and the
        execution result.

        - If operation is sent to a quantable platform, then its inputs and outputs
          will be quantized if necessary.
        - If operation is classified as shape-or-index related operation, then its
          execution will be taken with cpu.
        - If operation is sent to a fp32 platform, then its inputs and outputs shall
          never be quantized.

    ATTENTION:

        this dispatcher will insert necessary DeviceSwitch operations between
        shape-or-index operations and others.
    """

    def __init__(self, graph: BaseGraph) -> None:
        self.graph = graph

    def dispatch(
        self,
        quant_types: Collection[str],
        quant_platform: TargetPrecision = TargetPrecision.UNSPECIFIED,
        fp32_platform: TargetPrecision = TargetPrecision.FP32,
        soi_platform: TargetPrecision = TargetPrecision.SOI,
        **kwargs,
    ) -> Dict[str, TargetPrecision]:
        """Graph Dispatcher splits a graph into parts, each part of graph will
        be sent to a specific platform for further execution and quantization.

        There are 3 default platform during dispatching:
            quant_platform - all quantable parts of graph will be dispatched to this
                             platform
            SOI_platform   - Aka. Shape or Index related operations will be dispatched
                             to this platform.
            fp32_platform  - there are some operations receiving results from both
                             quant_platform and SOI_platform, they will be dispatched
                             they will be dispatched to fp32_platform.

        ATTENTION:

            Quantization follows this dispatching, and only the operations within
            quantable platform will be quantized in the future.

        ATTENTION:

            this dispatcher will insert necessary DeviceSwitch operations between
            shape-or-index operations and others.

        Args:
            graph (BaseGraph): graph object which going to be dispatched by this
                dispatcher.
            quant_types(Set[str]): all quantable types for given platforms.
            quant_platform (TargetPlatform):
                platform object where quantable parts will goes to.
            fp32_platform (TargetPlatform):
                platform object where SOI parts will goes to.
            SOI_platform (TargetPlatform):
                platform object where remaining parts will goes to.
        """
        graph = self.graph
        receivers, generators = soi_receivers(graph), soi_generators(graph)
        search_engine, soi_operations = SearchableGraph(graph), set(receivers)

        quant_operations = search_engine.opset_matching(
            sp_expr=lambda op: op.is_computing_op,
            rp_expr=value_tracing_pattern,
            ep_expr=lambda op: (op.type not in quant_types) or op.is_boundary,
            direction="down",
        )
        quant_operations.filter(lambda op: op.type not in quant_types)

        computing_extensions = search_engine.opset_matching(
            sp_expr=lambda op: op.is_computing_op,
            rp_expr=value_tracing_pattern,
            ep_expr=lambda op: (
                op.type in {"Shape", "TopK", "NonMaxSuppression"} or op.is_boundary
            ),
            direction="down",
        )

        # we assume all 'Shape', 'NonMaxSuppression', 'ConstantOfShape', 'Topk'
        # operations are SOI generators.
        shape_forward_matching = search_engine.opset_matching(
            sp_expr=lambda op: op in generators and op.type not in {"Constant"},
            rp_expr=value_tracing_pattern,
            ep_expr=lambda op: (
                op in receivers
                or op in quant_operations
                or op.is_boundary
                or op.is_computing_op
            ),
            direction="down",
        )

        # remove computing operations and quant operations from matching
        shape_forward_matching.filter(
            lambda op: op.is_computing_op or op in quant_operations
        )

        # update matchings, ready for further searching.
        soi_operations.update(shape_forward_matching)

        while True:
            # there are some particular cases where a single matching can not handle.
            # to cover all shape-related operations, a reverse matching is required.
            shape_backward_matching = search_engine.opset_matching(
                sp_expr=lambda op: op in soi_operations and op.type != "Shape",
                rp_expr=reverse_tracing_pattern,
                ep_expr=lambda op: (
                    op in soi_operations
                    or op in quant_operations
                    or op.is_boundary
                    or op.is_computing_op
                ),
                direction="up",
            )

            # remove computing operations and quant operations from matching
            shape_backward_matching.filter(
                lambda op: op.is_computing_op or op in quant_operations
            )

            if all([(op in soi_operations) for op in shape_backward_matching]):
                break

            # update matchings
            soi_operations.update(shape_backward_matching)

        # generate dispatching table.
        dispatching_table = {}
        for operation in graph.operations.values():
            if operation in soi_operations and operation not in computing_extensions:
                dispatching_table[operation.name] = soi_platform
            elif operation in quant_operations:
                dispatching_table[operation.name] = quant_platform
            else:
                dispatching_table[operation.name] = fp32_platform

        for operation in graph.operations.values():
            # move Topk, Shape, NonMaxSuppression to the platform same as their input.
            if operation.type in {"Shape", "TopK", "NonMaxSuppression"}:
                source_op = operation.inputs[0].source_op
                if source_op is not None:
                    dispatching_table[operation.name] = dispatching_table[
                        source_op.name
                    ]
                else:
                    dispatching_table[operation.name] = fp32_platform

            # move activations to the platform same as their input.
            if operation.is_linear_activation:
                source_op = operation.inputs[0].source_op
                if source_op is not None:
                    dispatching_table[operation.name] = dispatching_table[
                        source_op.name
                    ]

        return dispatching_table
