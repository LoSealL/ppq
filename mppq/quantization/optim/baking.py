from typing import Iterable, Optional

from mppq.defs import empty_ppq_cache
from mppq.executor.base import BaseGraphExecutor
from mppq.ir.base.graph import BaseGraph
from mppq.ir.base.quantize import QuantableOperation
from mppq.quantization.optim.base import OPTIM_ALGORITHMS, QuantizationOptimizationPass
from mppq.utils.qfunction import ppq_fake_quant


@OPTIM_ALGORITHMS.register()
class ParameterBakingPass(QuantizationOptimizationPass):
    r"""将计算图中所有已量化算子的参数进行烘焙。
    烘焙后的算子将省去伪量化的计算，可以加速约20%。

    Note:

        烘焙后的计算图不可逆
    """

    def __init__(self) -> None:
        super().__init__()
        self._quantize_function = ppq_fake_quant

    @empty_ppq_cache
    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Optional[Iterable] = None,
        executor: Optional[BaseGraphExecutor] = None,
        **kwargs,
    ):
        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation):
                continue
            operation.baking_parameters(self._quantize_function)
