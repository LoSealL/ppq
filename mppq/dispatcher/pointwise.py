"""
This File Contains Legacy Dispatchers.
Refer to ppq.scheduler.perseus for updated implementation.
"""

from typing import Collection, Dict

from mppq.dispatcher.base import DISPATCHER_TABLE
from mppq.dispatcher.conservative import ConservativeDispatcher
from mppq.quant import TargetPrecision


@DISPATCHER_TABLE.register("pointwise")
class PointDispatcher(ConservativeDispatcher):
    r"""Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.

    For the most part, all operations within graph can be partitioned into quantable
    operations, Shape-Or-Index related operations and remaining operations, all sub
    classes of GraphDispatcher will give an implementation of function "dispatch" to
    send all operations to their proper platform.

    Point Dispatch send all computing op to quantable platform, while other ops remain
    unchanged.

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

    def dispatch(
        self,
        quant_types: Collection[str],
        quant_precision: TargetPrecision,
        fp32_platform: TargetPrecision = TargetPrecision.FP32,
        soi_platform: TargetPrecision = TargetPrecision.SOI,
        **kwargs,
    ) -> Dict[str, TargetPrecision]:
        """Graph Dispatcher splits a graph into parts, each part of graph will
        be sent to a specific platform for further execution and quantization.

        There are 3 default platform during dispatching:
            quant_precision - all quantable parts of graph will be dispatched to this
                             platform
            SOI_platform   - Aka. Shape or Index related operations will be dispatched
                             to this platform.
            fp32_platform  - there are some operations receiving results from both
                             quant_precision and SOI_platform, they will be dispatched
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
            quant_precision (TargetPlatform): =
                platform object where quantable parts will goes to.
            SOI_platform (TargetPlatform):
                platform object where SOI parts will goes to.
            fp32_platform (TargetPlatform):
                platform object where remaining parts will goes to.
        """
        graph = self.graph

        dispatch_table = super().dispatch(
            quant_types=quant_types,
            quant_precision=quant_precision,
            fp32_platform=fp32_platform,
            soi_platform=soi_platform,
            kwargs=kwargs,
        )

        skip_ops = set()
        for op in graph.operations.values():
            if op in skip_ops:
                continue
            if op.type in quant_types and op.is_computing_op:
                dispatch_table[op.name] = quant_precision
            else:
                if dispatch_table[op.name] == quant_precision:
                    dispatch_table[op.name] = fp32_platform

        return dispatch_table
