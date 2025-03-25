"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import re
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

from mppq.ir.base.graph import BaseGraph
from mppq.ir.search import SearchableGraph
from mppq.logger import nest
from mppq.quant import TargetPrecision


@dataclass
class Subgraph:
    """Define a subgraph in ``IgnoredScope``."""

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


class IgnoredScope:
    r"""Create a scope to depict ignored quantizing operations in the graph.

    Args:
        types (List[str], optional): The types of operations to be ignored.
            Defaults to None.
        subgraphs (List[Subgraph | dict], optional): A subgraph which is
            defined by a list of input operations and output operations.
            Defaults to None.
        operations (List[str], optional): The names of operations to be ignored.
            The name can be either a fully matched string, a fuzzy string including
            wildcard characters ('*' and '?'), or a regular expression pattern
            which starts with "re://". Defaults to None.

    Example::

        model = dict(
            ignore=dict(
                type="IgnoredScope",
                types=["Conv", "Gemm"],
                operations=[r"re://.*conv.*", "Gemm?", "Concat_22"],
                subgraphs=[
                    dict(type="Subgraph", inputs=["input"], outputs=["output"])
                ],
            )
        )

    Note:

        Character "?" is a legal character in onnx name, if the op name contains '?', it
        will not be accurately matched, unless use the regular expression pattern.
        For example, to match the op name "Conv_123?split0", use the following string:
        r"re://Conv_123\?split0".
    """

    CASE_INSENSITIVE = False

    def __init__(
        self,
        types: Optional[List[str]] = None,
        subgraphs: Optional[List[Subgraph | dict]] = None,
        operations: Optional[List[str | re.Pattern]] = None,
    ):
        self.types = types or []
        self.subgraphs = subgraphs or []
        for i, subgraph in enumerate(self.subgraphs):
            if isinstance(subgraph, dict):
                self.subgraphs[i] = Subgraph(
                    inputs=subgraph.get("inputs", []),
                    outputs=subgraph.get("outputs", []),
                )
        self.operations = operations or []
        # normalize to regular expression
        for i, op in enumerate(self.operations):
            if isinstance(op, re.Pattern):
                continue
            if op.startswith("re://"):
                self.operations[i] = re.compile(op[5:])
            elif "*" in op or "?" in op:
                self.operations[i] = re.compile(op.replace("*", ".*").replace("?", "."))
        if IgnoredScope.CASE_INSENSITIVE and types:
            self.types = [t.lower() for t in types]
        self.log = nest(self.__class__.__name__)

    def dispatch(
        self, graph: BaseGraph, platform: TargetPrecision = TargetPrecision.FP32
    ) -> Dict[str, TargetPrecision]:
        """Dispatch the ignored scope to the graph.

        Args:
            graph (BaseGraph): The graph to apply the ignored scope to.
            platform (TargetPlatform, optional): Specify the platform for ignored
                operations. Defaults to TargetPlatform.FP32.
        """

        dispatch_table = {}
        for operation in graph.operations.values():
            op_type = operation.type
            if IgnoredScope.CASE_INSENSITIVE:
                op_type = operation.type.lower()
            op_name = operation.name
            if IgnoredScope.CASE_INSENSITIVE:
                op_name = operation.name.lower()
            if self.types and op_type in self.types:
                dispatch_table[operation.name] = platform
            if op_name in self.operations:
                dispatch_table[operation.name] = platform
            for p in filter(lambda x: isinstance(x, re.Pattern), self.operations):
                assert isinstance(p, re.Pattern)
                if p.fullmatch(op_name) is not None:
                    dispatch_table[operation.name] = platform

        def _op_match(op, pattern):
            return op.name in pattern

        for subgraph in self.subgraphs:
            assert isinstance(subgraph, Subgraph)
            operations = SearchableGraph(graph).opset_matching(
                sp_expr=partial(_op_match, pattern=subgraph.inputs),
                ep_expr=partial(_op_match, pattern=subgraph.outputs),
                rp_expr=lambda x, y: True,
                direction="down",
            )
            for operation in operations:
                dispatch_table[operation.name] = platform

        for operation in dispatch_table:
            self.log.debug(f"Ignored operations: {operation}")
        return dispatch_table
