from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, TypeVar

from .command import GraphCommand, GraphCommandType
from .graph import BaseGraph

T = TypeVar("T", BaseGraph, "GraphCommandProcessor")


class GraphCommandProcessor(metaclass=ABCMeta):
    r"""用于处理图上相关操作的抽象基类.

    我们使用指令-责任链模式处理 PPQ 计算图的相关操作，具体来说：

    所有图上相关操作都由一个 GraphCommand 对象进行封装，
    这一对象封装了操作的类型和必要参数

    同时我们设计了 GraphCommandProcessor 类用于接收并处理对应的 GraphCommand

    GraphCommandProcessor 被设计为责任链模式，当接收到无法处理的 GraphCommand 时
    将把无法识别 GraphCommand 传递给责任链上的下一任 GraphCommandProcessor
    直到 GraphCommand 被处理，或抛出无法处理的异常

    当你实现一个新的 GraphCommandProcessor 时，需要实现其中的方法 _acceptable_command_types，
    该方法返回了所有可以被识别的 GraphCommand 类型，同时在 _process 的逻辑中对 GraphCommand 的请求进行处理

    这两个方法被设计为私有的，这意味着你不能单独访问责任链中的独立 GraphCommandProcessor，
    只能够通过责任链的方式发起请求

    如果在责任链中有多个可以处理同一类请求的 GraphCommandProcessor，
    只有最先接触到 GraphCommand 的 GraphCommandProcessor将会被调用

    GraphCommandProcessor 将按照自定义逻辑解析 GraphCommand，
    在 BaseGraph 做出相应处理并给出结果，实现方法请参考 RunnableGraph

    Args:
        graph_or_processor (BaseGraph, Callable): 被处理的图对象，可以是 BaseGraph 或者
            GraphCommandProcessor. 如果是 GraphCommandProcessor 对象，则自动将 self 与 graph 链接成链
    """

    def __init__(self, graph_or_processor: T) -> None:
        self._next_command_processor: Optional[GraphCommandProcessor] = None
        if isinstance(graph_or_processor, GraphCommandProcessor):
            self._next_command_processor = graph_or_processor
            self._graph = graph_or_processor._graph
        else:
            self._graph = graph_or_processor

    @property
    @abstractmethod
    def _acceptable_command_types(self) -> List[GraphCommandType]:
        """Subclass of GraphCommandProcessor must give an implementation of
        this function.

            Return all acceptable GraphCommandTypes in a list as result.
            something like:
                return [
                    GraphCommandType.DEPLOY_TO_CPU,
                    GraphCommandType.DEPLOY_TO_CUDA,
                    GraphCommandType.DEPLOY_TO_NUMPY
                ]

        Returns:
            List[GraphCommandType]: all acceptable GraphCommandTypes
        """
        raise NotImplementedError

    def __call__(self, command: GraphCommand) -> Any:
        """Invoking interface of GraphCommandProcessor responsibility chain.
        All processors within the chain shall be invoked by this function one
        be one, until there is a processor claim to accept input command
        object, the entire processing of responsibility chain ends then.

        invoke a GraphCommandProcessor chain like that:

            _ = GraphCommandProcessor(graph, next_command_processor=None)
            _ = GraphCommandProcessor(graph, next_command_processor=_)
            command_processor = GraphCommandProcessor(graph, next_command_processor=_)

            command = GraphCommand(GraphCommandType.DEPLOY_TO_CUDA)
            command_processor(command)

        All three GraphCommandProcessor will then be called one by one

        Never attempt to use function like _(command) in above case,
        all responsibility chains should only be called by its head.

        Args:
            command (GraphCommand): An acceptable GraphCommand object
            if an improper GraphCommand is given, it will incurs ValueError at end.

        Raises:
            ValueError: raise when there is no suitable processor for your command.

        Returns:
            Any: processor will decide what is it result.
        """

        if not isinstance(command, GraphCommand):
            raise ValueError(
                "command should be an instance of GraphCommand,"
                f"however {type(command)} received yet."
            )

        if command.command_type in self._acceptable_command_types:
            return self.process(command)
        elif self._next_command_processor is not None:
            self._next_command_processor(command)
        else:
            raise ValueError(
                f"Command Type {command.command_type} is not acceptable in this graph, "
                "please make sure you have added proper command processor into "
                "processing chain.\n"
                f"For more information, you may refer to {self}"
            )

    @abstractmethod
    def process(self, command: GraphCommand) -> Any:
        """Subclass of GraphCommandProcessor must give an implementation of
        this function.

            Process received GraphCommand instance and give result(if there is any)

        Args:
            command (GraphCommand): input command object.

        Returns:
            Any: any result is fine.
        """
        raise NotImplementedError

    @property
    def graph(self) -> BaseGraph:
        return self._graph

    def __str__(self) -> str:
        return f"GraphCommandProcessor {self.__hash__}"
