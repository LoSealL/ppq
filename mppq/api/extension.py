"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Extend PPQ with customized platforms
"""

from copy import deepcopy
from typing import Dict, Optional, Type

from mppq.dispatcher.base import DISPATCHER_TABLE, GraphDispatcher
from mppq.executor.base import OPERATION_FORWARD_TABLE, BaseGraphExecutor
from mppq.executor.op.base import (
    ASSERT_IS_QUANT_OP,
    ASSERT_NUM_OF_INPUT,
    DEFAULT_BACKEND_TABLE,
    FORCE_CONVERT_DEVICE,
    GET_ATTRIBUTE_FROM_OPERATION,
    GET_VALUE_FROM_INPUTS,
    VALUE_TO_EXECUTING_DEVICE,
    OperationForwardProtocol,
)
from mppq.frontend import EXPORTER, PARSER
from mppq.frontend.onnx.onnx_exporter import OP_CONVERTERS
from mppq.ir.base.graph import GraphBuilder, GraphExporter, OperationExporter
from mppq.logger import error, warning
from mppq.quantization.optim.base import OPTIM_ALGORITHMS, QuantizationOptimizationPass
from mppq.quantizer.base import QUANTIZER, BaseQuantizer

_PLATFORM_TO_PARSER_: Dict[int, str] = {}
_PLATFORM_TO_EXPORTER_: Dict[int, str] = {}
_PLATFORM_TO_DISPATCHER_: Dict[int, str] = {}
_PLATFORM_TO_QUANTIZER_: Dict[int, str] = {}


def _register_operation_handler(
    handler: OperationForwardProtocol, operation_type: str, platform: int
):
    if platform not in OPERATION_FORWARD_TABLE:
        raise ValueError(
            "Unknown Platform detected, Please check your platform setting."
        )
    OPERATION_FORWARD_TABLE[platform].update({operation_type: handler})


def register_operation(op_type: str, platform: Optional[int] = None):
    r"""Register a new operation to forward table.

    Function should accept at least 3 input parameters, return one or more tensor
    as results:

    .. code-block:: python

        def func(
            op: Operation,
            values: List[torch.Tensor],
            ctx: Optional[TorchBackendContext] = None,
            **kwargs,
        ) -> torch.Tensor | Tuple[torch.Tensor,...]:
            ...

    If there is already another operation handler for given operation_type,
        new handler will replace the old one without warning.

    Args:
        op_type (str): name of the operation type
        platform (TargetPlatform, optional): specify a platform to register, if
            platform is None, add to the default backend table. Defaults to None.
    """

    def _wrapper(func: OperationForwardProtocol) -> OperationForwardProtocol:
        if platform is None:
            for i in OPERATION_FORWARD_TABLE.keys():
                _register_operation_handler(func, op_type, i)
        else:
            _register_operation_handler(func, op_type, platform)

        return func

    return _wrapper


def register_platform(
    platform_id: int,
    dispatcher: Dict[str | None, Optional[Type[GraphDispatcher]]],
    quantizer: Dict[str | None, Optional[Type[BaseQuantizer]]],
    parsers: Optional[Dict[str | None, Type[GraphBuilder]]] = None,
    exporters: Optional[Dict[str | None, Type[GraphExporter]]] = None,
):
    r"""注册一个新的自定义平台。

    Args:
        platform_id (int): 自定义平台的 ID ，不可重复。
        dispatcher (Dict[str | None, Type[GraphDispatcher]]): 自定义平台的调度器。
            字典键值可选，作为调度器的命名，和调度器的类类型。
        quantizer (Dict[str | None, Type[BaseQuantizer]]): 自定义平台的量化器。
            字典键值可选，作为量化器的命名，和量化器的类类型。
        parsers (Optional[Dict[str | None, Type[GraphBuilder]]], optional):
            自定义平台的图构建器。字典键值可选，作为图构建器的命名，和图构建器的类类型。
            Defaults to None.
        exporters (Optional[Dict[str | None, Type[GraphExporter]]], optional):
            自定义平台的图导出器。字典键值可选，作为图导出器的命名，和图导出器的类类型。
            Defaults to None.
    """
    if platform_id in _PLATFORM_TO_DISPATCHER_:
        raise KeyError(f"Platform {platform_id} is already registered.")
    OPERATION_FORWARD_TABLE[platform_id] = deepcopy(DEFAULT_BACKEND_TABLE)

    if parsers is not None:
        for i, (name, parser_cls) in enumerate(parsers.items()):
            if (name or parser_cls) in PARSER:
                warning(f"Parser {name} is replaced by {parser_cls}")
            PARSER.register(name)(parser_cls)
            name = PARSER.query_name(parser_cls)
            if i > 0:
                warning(
                    f"more than 1 parser is registered to platform {platform_id}, "
                    f"by default the last one is used. Current parser is {name}"
                )
            _PLATFORM_TO_PARSER_[platform_id] = name
    else:
        _PLATFORM_TO_PARSER_[platform_id] = "onnx"

    if exporters is not None:
        for i, (name, exporter_cls) in enumerate(exporters.items()):
            if (name or exporter_cls) in EXPORTER:
                warning(f"Exporter {name} is replaced by {exporter_cls}")
            EXPORTER.register(name)(exporter_cls)
            name = EXPORTER.query_name(exporter_cls)
            if i > 0:
                warning(
                    f"more than 1 exporter is registered to platform {platform_id}, "
                    f"by default the last one is used. Current exporter is {name}"
                )
            _PLATFORM_TO_EXPORTER_[platform_id] = name
    else:
        _PLATFORM_TO_EXPORTER_[platform_id] = "onnx"

    for i, (name, dispatcher_cls) in enumerate(dispatcher.items()):
        assert name is not None or dispatcher_cls is not None
        if (name or dispatcher_cls) in DISPATCHER_TABLE and dispatcher_cls is not None:
            warning(f"Dispatcher {name} is replaced by {dispatcher_cls}")
        if dispatcher_cls is not None:
            DISPATCHER_TABLE.register(name)(dispatcher_cls)
            name = DISPATCHER_TABLE.query_name(dispatcher_cls)
        elif not name:
            error("Both name and dispatcher are None.")
            raise ValueError
        if i > 0:
            warning(
                f"more than 1 dispatcher is registered to platform {platform_id}, "
                f"by default the last one is used. Current dispatcher is {name}"
            )
        _PLATFORM_TO_DISPATCHER_[platform_id] = name

    _PLATFORM_TO_QUANTIZER_[platform_id] = "default"
    for i, (name, quantizer_cls) in enumerate(quantizer.items()):
        if (name or quantizer_cls) in QUANTIZER and quantizer_cls is not None:
            warning(f"Quantizer {name} is replaced by {quantizer_cls}")
        if quantizer_cls is not None:
            QUANTIZER.register(name)(quantizer_cls)
            name = QUANTIZER.query_name(quantizer_cls)
        elif not name:
            error("Both name and quantizer are None.")
            raise ValueError
        if i > 0:
            warning(
                f"more than 1 quantizer is registered to platform {platform_id}, "
                f"by default the last one is used. Current quantizer is {name}"
            )
        _PLATFORM_TO_QUANTIZER_[platform_id] = name


__all__ = [
    "register_operation",
    "register_platform",
    # registry
    "DISPATCHER_TABLE",
    "PARSER",
    "EXPORTER",
    "QUANTIZER",
    "OP_CONVERTERS",
    "OPTIM_ALGORITHMS",
    # base classes
    "GraphDispatcher",
    "GraphBuilder",
    "GraphExporter",
    "OperationExporter",
    "BaseQuantizer",
    "BaseGraphExecutor",
    "QuantizationOptimizationPass",
    # extension api
    "ASSERT_IS_QUANT_OP",
    "ASSERT_NUM_OF_INPUT",
    "FORCE_CONVERT_DEVICE",
    "GET_ATTRIBUTE_FROM_OPERATION",
    "GET_VALUE_FROM_INPUTS",
    "VALUE_TO_EXECUTING_DEVICE",
]
