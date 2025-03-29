from mppq.frontend.base import EXPORTER, PARSER
from mppq.frontend.onnx.onnx_exporter import OnnxExporter
from mppq.frontend.onnx.onnx_parser import OnnxParser
from mppq.frontend.onnx.onnxruntime_exporter import ONNXRUNTIMExporter
from mppq.frontend.onnx.openvino_exporter import OpenvinoExporter

__all__ = [
    "OnnxExporter",
    "OnnxParser",
    "ONNXRUNTIMExporter",
    "OpenvinoExporter",
]


PARSER.register("onnx")(OnnxParser)
EXPORTER.register("onnx")(OpenvinoExporter)
