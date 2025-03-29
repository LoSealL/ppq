"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

PPQ API.

Users are always encouraged to use functions and object within this package.
推荐用户仅使用该包内的函数和对象。
"""

from mppq import __version__
from mppq.api.extension import register_operation, register_platform
from mppq.api.interface import (
    dispatch_graph,
    export_graph,
    export_onnx_graph,
    format_graph,
    load_graph,
    load_onnx_graph,
    load_quantizer,
    quantize,
)
from mppq.dispatcher.base import GraphDispatcher
from mppq.executor.torch import TorchExecutor
from mppq.ffi import ENABLE_CUDA_KERNEL
from mppq.frontend.base import GraphBuilder, GraphExporter
from mppq.logger import set_level as set_log_level
from mppq.quantizer.base import BaseQuantizer

__all__ = [
    "__version__",
    "TorchExecutor",
    "ENABLE_CUDA_KERNEL",
    "set_log_level",
    "register_operation",
    "register_platform",
    "dispatch_graph",
    "export_graph",
    "export_onnx_graph",
    "format_graph",
    "load_graph",
    "load_onnx_graph",
    "load_quantizer",
    "quantize",
    "GraphDispatcher",
    "GraphBuilder",
    "GraphExporter",
    "BaseQuantizer",
]

print(
    r"""
              ____  ____  ____
   ____ ___  / __ \/ __ \/ __ \
  / __ `__ \/ /_/ / /_/ / / / /
 / / / / / / ____/ ____/ /_/ /
/_/ /_/ /_/_/   /_/    \___\_\

"""
)
