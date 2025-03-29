"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Changelist:
- Fix: build variables with correct data type.
- Feat: set known shape on building variables.
"""

# pylint: disable=protected-access

import os
from contextlib import suppress
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import numpy as np
import onnx
import onnx.version_converter as ovc
from onnx import helper, mapping, numpy_helper

from mppq.common import (
    DEFAULT_OPSET_DOMAIN,
    DEFAULT_OPSET_VERSION,
    GRAPH_OPSET_ATTRIB,
    ONNX_VERSION,
)
from mppq.data import DataType
from mppq.ir.base.graph import BaseGraph, GraphBuilder
from mppq.ir.base.opdef import Operation, Opset, Variable
from mppq.logger import nest


class OnnxParser(GraphBuilder):
    """Parse ONNX model to PPQ IR graph."""

    _log = nest("OnnxParser")

    def _build_variables(
        self,
        graph: BaseGraph,
        graph_inputs: List[str],
        graph_outputs: List[str],
        op_inputs: Dict[str, list],
        op_outputs: Dict[str, list],
        value_info: Dict[str, SimpleNamespace],
    ) -> BaseGraph:
        var_list = []

        for op_name, _ in graph.operations.items():
            for var_name in op_inputs[op_name]:
                var_list.append(var_name)
            for var_name in op_outputs[op_name]:
                var_list.append(var_name)

        # create all variable at once.
        for var_name in set(var_list):
            if var_name in value_info:
                var_dtype = DataType.from_numpy(value_info[var_name].dtype)
                var_shape = list(value_info[var_name].shape)
            else:
                var_dtype = DataType.FP32
                var_shape = None
            graph.variables[var_name] = Variable(
                name=var_name, dtype=var_dtype, shape=var_shape
            )

        # build graph's input, output variables.
        try:
            for var_name in graph_inputs:
                if var_name not in graph.variables:
                    continue
                graph.inputs[var_name] = graph.variables[var_name]
            for var_name in graph_outputs:
                graph.outputs[var_name] = graph.variables[var_name]
        except KeyError as e:
            raise KeyError(
                "seems you got an input/output variable that is not linked to any "
                "operation."
            ) from e

        # build operation inputs, outputs variables.
        for op in graph.operations.values():
            for var_name in op_inputs[op.name]:
                var = graph.variables[var_name]
                var.dest_ops.append(op)
                op.inputs.append(graph.variables[var_name])
            for var_name in op_outputs[op.name]:
                var = graph.variables[var_name]
                var.source_op = op
                op.outputs.append(graph.variables[var_name])
        return graph

    def _initialize_params(
        self, graph: BaseGraph, initializer: Dict[str, Any]
    ) -> BaseGraph:
        for var in graph.variables.values():
            if var.name in initializer:
                for dest_op in var.dest_ops:
                    assert isinstance(dest_op, Operation)
                    dest_op.parameters.append(var)
                var.value = initializer[var.name]
                var.is_parameter = True
        return graph

    def _de_inplace(self, graph: BaseGraph) -> BaseGraph:
        """Remove inplace layer in netdef if the names of bottom and top are same,
        it means the computation of this layer is in place."""

        def new_name(_name: str):
            if _name == "":
                return ""
            elif _name not in total_write_times or current_write_times:
                return _name
            elif current_write_times[_name] == total_write_times[_name]:
                return _name
            else:
                return f"{_name}_ver{current_write_times[_name]}"

        total_write_times: Dict[str, int] = {}
        for op in graph.operations.values():
            for top in op.outputs:
                total_write_times.setdefault(top.name, 0)
                total_write_times[top.name] += 1

        current_write_times = {}
        for name in graph.inputs.keys():
            total_write_times[name] = 0
            current_write_times[name] = 0

        for op in graph.operations.values():
            for bottom in op.inputs:
                if bottom.is_parameter:
                    continue
                bottom.name = new_name(bottom.name)
            for top in op.outputs:
                current_write_times.setdefault(top.name, 0)
                current_write_times[top.name] += 1
                top.name = new_name(top.name)
        return graph

    def _refine_graph(self, graph: BaseGraph) -> BaseGraph:
        for op in graph.operations.values():
            for key, value in op.attributes.items():
                if op.type == "Constant" or op.type == "ConstantOfShape":
                    # The attribute of 'Constant' node is a value, needs to convert to
                    # numpy array
                    assert isinstance(value, onnx.TensorProto)
                    value = numpy_helper.to_array(value).copy()
                if op.type == "Cast":
                    # The attribute of 'Cast' node is data type (represented in int),
                    # need to convert to numpy data type
                    assert isinstance(value, int)
                    value = helper.tensor_dtype_to_np_dtype(value)
                elif isinstance(value, bytes):
                    value = value.decode("utf-8")
                op.attributes[key] = value

        for input_var in list(graph.inputs.values()):
            # remove initializer from graph.inputs
            if input_var.has_value:
                graph.inputs.pop(input_var.name)
        return graph

    def _convert_opsets_to_str(self, opsets: Iterable) -> List[Dict[str, str]]:
        results = []
        for opset in opsets:
            results.append({"domain": opset.domain, "version": opset.version})
        return results

    def _build_graph(
        self,
        model_pb: onnx.ModelProto,
        opset: int,
        graph: BaseGraph,
        op_inputs_dict: Dict[str, List[str]],
        op_outputs_dict: Dict[str, List[str]],
    ):
        _rand_seed = 0  # used for name generation.
        for node in model_pb.graph.node:
            op_name = node.name
            if len(op_name) == 0:
                # some operation do not have a name, we just generate one.
                op_name = "generated_name_" + str(_rand_seed)
                _rand_seed += 1

            if op_name in graph.operations:
                raise KeyError(f"Duplicated operation {op_name} was found.")

            graph.operations[op_name] = Operation(
                name=op_name,
                op_type=node.op_type,
                attributes={
                    item.name: helper.get_attribute_value(item)
                    for item in node.attribute
                },
                opset=Opset(domain=DEFAULT_OPSET_DOMAIN, version=opset),
            )
            op_inputs_dict[op_name] = [var_name for var_name in node.input]
            op_outputs_dict[op_name] = [var_name for var_name in node.output]

    def build(
        self, model_object: str | os.PathLike | onnx.ModelProto, **kw
    ) -> BaseGraph:
        """Build PPQ IR graph from an onnx file."""
        if kw:
            self._log.warning("Extra argument is not accepted! Ignoring:")
            self._log.warning(f'\n  {"\n  ".join(kw.keys())}')
        if isinstance(model_object, onnx.ModelProto):
            model_pb = model_object
        elif not Path(model_object).exists():
            self._log.error(f"File {model_object} does not exist.")
            raise FileNotFoundError
        else:
            model_pb = onnx.load_model(model_object, load_external_data=False)
        if model_pb.ir_version < ONNX_VERSION:
            # try to upgrade onnx version to the latest
            with suppress(ovc.ConvertError):
                model_pb = ovc.convert_version(model_pb, DEFAULT_OPSET_VERSION)

        onnx.checker.check_model(model_pb)  # validate model
        opsets = model_pb.opset_import
        graph = BaseGraph(name=model_pb.graph.name)
        graph._detail[GRAPH_OPSET_ATTRIB] = self._convert_opsets_to_str(opsets)
        graph._detail["ir_version"] = model_pb.ir_version
        with suppress(onnx.shape_inference.InferenceError):
            model_pb = onnx.shape_inference.infer_shapes(model_pb, True)

        onnx_import_opset = DEFAULT_OPSET_VERSION
        for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
            if opset["domain"] == DEFAULT_OPSET_DOMAIN or opset["domain"] == "":
                onnx_import_opset = opset["version"]
                break

        # a temporary storage for operation's inputs and outputs
        op_inputs_dict: Dict[str, List[str]] = {}
        op_outputs_dict: Dict[str, List[str]] = {}
        self._build_graph(
            model_pb, onnx_import_opset, graph, op_inputs_dict, op_outputs_dict
        )
        # value type info
        graph_pb = model_pb.graph
        value_info = {}
        for value in chain(graph_pb.input, graph_pb.output, graph_pb.value_info):
            if value.name in value_info:
                # this is not an error since input/output can be appeared in value info
                continue
            value_shape = [i.dim_value for i in value.type.tensor_type.shape.dim]
            value_dtype = mapping.TENSOR_TYPE_MAP[value.type.tensor_type.elem_type]
            value_info[value.name] = SimpleNamespace(
                shape=value_shape, dtype=value_dtype.np_dtype
            )

        initializer = {}
        for item in graph_pb.initializer:
            init_name = item.name
            value = numpy_helper.to_array(item)
            if value.dtype == np.float16:
                # we should cast half type back to float32 because most cpu kernels
                # don't support half type.
                value = value.astype("float32")
            initializer[init_name] = value

        inputs = [item.name for item in graph_pb.input]
        outputs = [item.name for item in graph_pb.output]
        graph = self._build_variables(
            graph,
            graph_inputs=inputs,
            graph_outputs=outputs,
            op_inputs=op_inputs_dict,
            op_outputs=op_outputs_dict,
            value_info=value_info,
        )
        graph = self._initialize_params(graph, initializer)
        graph = self._de_inplace(graph)
        graph = self._refine_graph(graph)
        return graph
