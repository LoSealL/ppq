from abc import ABCMeta, abstractmethod
from typing import Iterable, List, Optional, Self, Sequence, Type

from mppq.executor.base import BaseGraphExecutor
from mppq.ir.base.graph import BaseGraph
from mppq.logger import error, info
from mppq.register import Registry


class QuantizationOptimizationPass(metaclass=ABCMeta):
    r"""QuantizationOptimizationPass is a basic building block of PPQ
    quantization logic.

    PPQ is designed as a Multi pass Compiler of quantization network.
        where pass here refers to a traversal through the entire network.

    This class is an abstract base class of all customized passes.
    Quantizer will build an optimization pipeline later to quantize and optimize
    your network.
    """

    def __init__(self, name: Optional[str] = None, **kw) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Optional[Iterable] = None,
        executor: Optional[BaseGraphExecutor] = None,
        **kwargs,
    ) -> Optional[BaseGraph]:
        """Apply an optimization algorithm to a given graph."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"QuantizationOptimizationPass[{self.name}]"


class QuantizationOptimizationPipeline(QuantizationOptimizationPass):
    """QuantizationOptimizationPipeline is a sorted set PPQ Optimization
    passes.

    PPQ is designed as a Multi pass Compiler of quantization network.
        where pass here refers to a traversal through the entire network.

    Quantizer is going to calling optimization pass from pipeline one by one to
        eventually finish network quantization procedure
    """

    def __init__(self, passes: Sequence[QuantizationOptimizationPass]):
        super().__init__(name="Quantization Optimization Pipeline")
        self._pipeline: List[QuantizationOptimizationPass] = []
        for optim in passes:
            self.append(optim)

    def __len__(self) -> int:
        return len([_ for _ in self.__iter__()])

    def __contains__(self, obj: QuantizationOptimizationPass) -> bool:
        return obj in self._pipeline

    def __iter__(self) -> Iterable[QuantizationOptimizationPass]:
        for i in self._pipeline:
            if isinstance(i, QuantizationOptimizationPipeline):
                # support nested pipeline
                yield from i.__iter__()
            else:
                yield i

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Optional[Iterable] = None,
        executor: Optional[BaseGraphExecutor] = None,
        verbose: bool = True,
        **kwargs,
    ):
        for i, optim in enumerate(self.__iter__()):
            assert isinstance(optim, QuantizationOptimizationPass)
            if verbose:
                info(f"[{i:02d}/{len(self):02d} {optim.name} Running ...")

            if not isinstance(graph, BaseGraph):
                raise TypeError(
                    "parameter 1 should be an instance of PPQ BaseGraph when "
                    f"calling optim pass, however {type(graph)} was given."
                )
            try:
                ret = optim.optimize(
                    graph=graph,
                    dataloader=dataloader,
                    executor=executor,
                    verbose=verbose,
                    **kwargs,
                )
            except Exception:
                error(f"An error occurred while running {optim.name}.")
                raise
            finally:
                if verbose:
                    info(f"{optim.name} Finished.")
            if isinstance(ret, BaseGraph):
                graph = ret
        return graph

    def append(self, optim: QuantizationOptimizationPass) -> Self:
        """Add an optimization pass to the end of pipeline."""
        if not isinstance(optim, QuantizationOptimizationPass):
            error("Pass should be a subclass of QuantizationOptimizationPass")
            error(f"Unexpected pass type: {type(optim)}")
            raise TypeError
        self._pipeline.append(optim)
        return self

    def prepend(self, optim: QuantizationOptimizationPass) -> Self:
        """Add an optimization pass to the beginning of pipeline."""
        if not isinstance(optim, QuantizationOptimizationPass):
            error("Pass should be a subclass of QuantizationOptimizationPass")
            error(f"Unexpected pass type: {type(optim)}")
            raise TypeError
        self._pipeline = [optim] + self._pipeline
        return self

    def __repr__(self) -> str:
        report = ""
        for optimization_pass in self._pipeline:
            report += str(optimization_pass) + "\n"
        return report


OPTIM_ALGORITHMS: Registry[Type[QuantizationOptimizationPass]] = Registry(
    "OPTIM_ALGORITHMS"
)
