"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

PPQ frontend parser.

Currently support parser:
- onnx
- onnxruntime (onnx with Microsoft domain operators)
- openvino
"""

from importlib import import_module
from pathlib import Path

from .base import EXPORTER, PARSER

# auto scan directory and load
for folder in Path(__file__).parent.iterdir():
    if folder.is_dir() and not folder.name.startswith("_"):
        import_module(f".{folder.name}", package=__name__)


__all__ = ["EXPORTER", "PARSER"]
