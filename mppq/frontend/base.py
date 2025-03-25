"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from typing import Type

from mppq.ir.base.graph import GraphBuilder, GraphExporter
from mppq.register import Registry

PARSER: Registry[Type[GraphBuilder]] = Registry("PARSER")
EXPORTER: Registry[Type[GraphExporter]] = Registry("EXPORTER")
