from mppq.defs import empty_ppq_cache
from mppq.ir.base.graph import BaseGraph
from mppq.ir.base.quantize import QuantableOperation
from mppq.quantization.qfunction import PPQuantFunction

from .base import OPTIM_ALGORITHMS, QuantizationOptimizationPass


@OPTIM_ALGORITHMS.register()
class ParameterBakingPass(QuantizationOptimizationPass):
    r"""将计算图中所有已量化算子的参数进行烘焙。
    烘焙后的算子将省去伪量化的计算，可以加速约20%。

    Note:

        烘焙后的计算图不可逆
    """

    def __init__(self) -> None:
        super().__init__()
        self._quantize_function = PPQuantFunction

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs):
        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation):
                continue
            operation.baking_parameters(self._quantize_function)
