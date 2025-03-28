"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import os
from typing import Optional, Union

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor,
    make_tensor_value_info,
)

import mppq
from mppq.common import GRAPH_OPSET_ATTRIB, ONNX_EXPORT_OPSET, ONNX_VERSION
from mppq.data import DataType, convert_any_to_numpy
from mppq.ir.base.graph import BaseGraph, GraphExporter, OperationExporter
from mppq.ir.base.opdef import Operation, Variable


class ConstantOfShapeExporter(OperationExporter):
    """
    PATCH 20211203, ConstantOfShape Op causes an export error.
    这一问题是由 ConstantOfShape 中的 value 格式问题引发的，下面的代码将导出正确的格式
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        op.attributes["value"] = numpy_helper.from_array(op.attributes["value"])
        return op


class MMCVExporter(OperationExporter):
    """mmcv operation must have a domain attribute."""

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        op.attributes["domain"] = "mmcv"
        return op


class InterpExporter(OperationExporter):
    """
    PATCH 20211216, interp op can not export input_shape attribute.
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        op.attributes.pop("input_shape")
        return op


class ORTExporter(OperationExporter):
    """onnxruntime operation must have a domain attribute."""

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        op.attributes["domain"] = "com.microsoft"
        return op


OP_CONVERTERS = {
    "ConstantOfShape": ConstantOfShapeExporter,
    "MMCVRoiAlign": MMCVExporter,
    "grid_sampler": MMCVExporter,
    "Interp": InterpExporter,
    "Attention": ORTExporter,
    "QAttention": ORTExporter,
    "QGemm": ORTExporter,
    "QLinearAdd": ORTExporter,
    "QLinearAveragePool": ORTExporter,
    "QLinearConcat": ORTExporter,
    "QLinearConv": ORTExporter,
    "QLinearGlobalAveragePool": ORTExporter,
    "QLinearLeakyRelu": ORTExporter,
    "QLinearMul": ORTExporter,
    "QLinearReduceMean": ORTExporter,
    "QLinearSigmoid": ORTExporter,
}


def _convert_value(value: Union[int, float, np.ndarray, torch.Tensor]):
    if isinstance(value, (int, float)):
        return value
    else:
        x = convert_any_to_numpy(value, accept_none=True)
        if x is None:
            return
        return x.tolist()


class OnnxExporter(GraphExporter):
    """
    PPQ 可以将 计算图 导出成 Onnx 标准格式，Onnx Exporter 不会导出 QDQ 节点。
    如需导出带有 QDQ 节点的 Onnx 文件，用户需要使用 OnnxRuntime Exporter

    任何导出器的导出逻辑都是原地进行的，它们将对传入的计算图对象进行原地修改，因此在导出之前你需要手动克隆计算图。
    """

    def export_graph(self, graph: BaseGraph) -> onnx.ModelProto:
        """
        Convert a PPQ IR to Onnx IR.
        This export will only convert PPQ Op and var to onnx, all quantization configs
        will be skipped.

        This function will try to keep the opset version of your graph unchanged.
        However if the opset is not given, ppq will convert it to with the global
        parameter ppq.core.ONNX_EXPORT_OPSET.
        """

        name = graph.name or f"mppq - v({mppq.__version__})"
        # Ready to export onnx graph definition.
        _inputs, _outputs, _initilizers, _nodes, _value_info = [], [], [], [], []

        # before we can export them, we firstly convert all ops to proper format.
        for op in graph.topological_sort():
            if op.type in OP_CONVERTERS:
                exporter = OP_CONVERTERS[op.type]()
                assert isinstance(
                    exporter, OperationExporter
                ), f"Expected an OpExporter here, however {type(exporter)} was given."
                op = exporter.export(op=op, graph=graph)

        for op in graph.topological_sort():
            _nodes.append(self.build_operator_proto(op))

        for variable in graph.variables.values():
            if variable.name in graph.inputs:
                var_shape = variable.shape
                onnx_dtype = variable.dtype.value
                _inputs.append(
                    make_tensor_value_info(
                        name=variable.name, elem_type=onnx_dtype, shape=var_shape
                    )
                )
            if variable.name in graph.outputs:
                var_shape = variable.shape
                onnx_dtype = variable.dtype.value
                _outputs.append(
                    make_tensor_value_info(
                        name=variable.name, elem_type=onnx_dtype, shape=var_shape
                    )
                )
            else:
                tensor_proto = self.build_variable_proto(variable)
                if not tensor_proto:
                    continue
                if variable.is_parameter:
                    _initilizers.append(tensor_proto)
                else:
                    _value_info.append(tensor_proto)

        graph_def = make_graph(
            name=name,
            nodes=_nodes,
            inputs=_inputs,
            outputs=_outputs,
            initializer=_initilizers,
            value_info=_value_info,
        )

        # if opset is missing from your graph, give it a default one.
        if GRAPH_OPSET_ATTRIB not in graph.extension_attrib:
            opsets = [make_operatorsetid("", ONNX_EXPORT_OPSET)]
        else:
            opsets = []
            for opset in graph.extension_attrib[GRAPH_OPSET_ATTRIB]:
                opsets.append(make_operatorsetid(opset["domain"], opset["version"]))
        onnx_model = make_model(
            graph_def,
            producer_name="mppq",
            ir_version=graph.extension_attrib.get("ir_version", ONNX_VERSION),
            opset_imports=opsets,
        )
        return onnx_model

    def build_operator_proto(self, operation: Operation) -> onnx.NodeProto:
        """
        Convert PPQ Op to Onnx Operation
        An Op consumes zero or more Tensors, and produces zero or more Tensors.
        """
        attributes = operation.attributes
        for key in attributes:
            value = attributes[key]
            if isinstance(value, DataType):
                attributes[key] = value.value
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    attributes[key] = None
                elif value.numel() == 1:
                    # convert to 1d array
                    attributes[key] = convert_any_to_numpy([value.item()])
                else:
                    attributes[key] = convert_any_to_numpy(value)

        op_proto = make_node(
            op_type=operation.type,
            inputs=[_.name for _ in operation.inputs],
            outputs=[_.name for _ in operation.outputs],
            name=operation.name,
            **attributes,
        )

        return op_proto

    def build_variable_proto(self, variable: Variable) -> onnx.TensorProto | None:
        """
        Convert PPQ Variable to Onnx TensorProto, There are 2 different types of Tensor
        in Onnx:
            Variable: Represents a Tensor whose value is not known until inference-time.
            Constant: Represents a Tensor whose value is known.
        """
        # Parameter Variable in PPQ, Constant Variable in Onnx
        if not variable.has_value:
            return

        var_shape = variable.value.shape
        pytorch_dtype = variable.value.dtype
        onnx_dtype = DataType.from_torch(pytorch_dtype).value
        value = variable.value
        is_raw_format = False
        if value.numel() == 0:
            value = []
        elif value.ndim == 0:  # Pytorch Scalar Type
            value = [value.item()]
        elif value.ndim >= 1:
            value = convert_any_to_numpy(value, False).flatten()
            value = value.tobytes()
            is_raw_format = True
        tensor_proto = make_tensor(
            name=variable.name,
            data_type=onnx_dtype,
            dims=var_shape,
            vals=value,
            raw=is_raw_format,
        )
        return tensor_proto

    def export(
        self,
        file_path: str | os.PathLike,
        graph: BaseGraph,
        config_path: Optional[str] = None,
        save_as_external_data: bool = False,
        **kwargs,
    ):
        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            self.export_quantization_config(config_path, graph)

        size_threshold = 0 if save_as_external_data else 1024
        model_pb = self.export_graph(graph=graph)
        onnx.save_model(
            model_pb,
            file_path,
            save_as_external_data=save_as_external_data,
            size_threshold=size_threshold,
            convert_attribute=True,
        )
