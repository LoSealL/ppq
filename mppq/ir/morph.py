from typing import Any, List, Optional

import torch

from mppq.data import (
    DataType,
    convert_any_to_python_primary_type,
    convert_any_to_tensor,
)
from mppq.ir.base.command import (
    GraphCommand,
    GraphCommandType,
    ReplaceOperationCommand,
    ReplaceVariableCommand,
    TruncateGraphCommand,
)
from mppq.ir.base.graph import Operation, Variable
from mppq.ir.base.processor import GraphCommandProcessor
from mppq.ir.search import SearchableGraph
from mppq.logger import error, warning


class GraphReplacer(GraphCommandProcessor):
    r"""Graph Replacer offers a bunch of graph editing functions that helps
    replacing operation or variable in your graph.
    """

    def process(self, command: GraphCommand) -> Any:
        if command.command_type == GraphCommandType.REPLACE_OP:
            assert isinstance(
                command, ReplaceOperationCommand
            ), "Use ReplaceOperationCommand instead of GraphCommand"
            return self.replace_op(command.op_name, command.replace_to)
        if command.command_type == GraphCommandType.REPLACE_VAR:
            assert isinstance(
                command, ReplaceVariableCommand
            ), "Use ReplaceOperationCommand instead of GraphCommand"
            return self.replace_var(command.op_name, command.replace_to)
        if command.command_type == GraphCommandType.REPLACE_BATCHNORM_TO_CONV:
            return self.replace_batchnorm_to_conv()
        if command.command_type == GraphCommandType.REPLACE_BATCHNORM_TO_SCALE:
            return self.replace_batchnorm_to_scale()

    def replace_op(self, op_name: str, replace_to: Operation):
        if op_name not in self._graph.operations:
            error(f"Operation {op_name} is not in current graph")
            raise KeyError
        operation = self._graph.operations[op_name]

        replace_to.inputs.clear()
        replace_to.inputs.extend(operation.inputs)
        for input_var in operation.inputs:
            dest_idx = input_var.dest_ops.index(operation)
            input_var.dest_ops[dest_idx] = replace_to

        replace_to.outputs.clear()
        replace_to.outputs.extend(operation.outputs)
        for output_var in operation.outputs:
            output_var.source_op = replace_to

        replace_to.parameters.clear()
        replace_to.parameters.extend(operation.parameters)

        self._graph.operations[op_name] = replace_to

    def replace_var(self, var_name: str, replace_to: Variable):
        if var_name not in self._graph.variables:
            error(f"Variable {var_name} is not in current graph")
            raise KeyError
        variable = self._graph.variables[var_name]

        replace_to.dest_ops.clear()
        replace_to.dest_ops.extend(variable.dest_ops)
        for dest_op in replace_to.dest_ops:
            dest_idx = dest_op.inputs.index(variable)
            dest_op.inputs[dest_idx] = replace_to

        replace_to.source_op = variable.source_op
        if variable.source_op is not None:
            source_idx = variable.source_op.outputs.index(variable)
            variable.source_op.outputs[source_idx] = replace_to

        self._graph.variables[var_name] = replace_to
        if var_name in self._graph.inputs:
            self._graph.inputs[var_name] = replace_to
        if var_name in self._graph.outputs:
            self._graph.outputs[var_name] = replace_to

    @property
    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            GraphCommandType.REPLACE_OP,
            GraphCommandType.REPLACE_VAR,
            GraphCommandType.REPLACE_BATCHNORM_TO_CONV,
            GraphCommandType.REPLACE_BATCHNORM_TO_SCALE,
        ]

    def replace_batchnorm_to_conv(self, dimension: int = 2):
        """Replace Batchnorm to 1D Convolution."""
        for op in self.graph.operations.values():
            if op.type == "BatchNormalization":
                warning(
                    f"Isolated BatchNormalization({op.name}) was detected, "
                    f"PPQ will replace it to 1*1 Convolution({dimension}D)."
                )

                assert (
                    len(op.parameters) == 4
                ), "BatchNorm should have 4 parameters, namely alpha, beta, mean, var"
                alpha = op.parameters[0].value
                beta = op.parameters[1].value
                mean = op.parameters[2].value
                var = op.parameters[3].value
                epsilon = op.attributes.get("epsilon", 1e-5)

                with torch.no_grad():
                    w = alpha / torch.sqrt(var + epsilon)
                    w = w.reshape([-1, 1] + [1] * dimension)
                    b = alpha * (-mean) / torch.sqrt(var + epsilon) + beta

                op.type = "Conv"
                op.attributes.clear()
                op.attributes["kernel_shape"] = [1] * dimension
                op.attributes["strides"] = [1] * dimension
                op.attributes["dilations"] = [1] * dimension
                op.attributes["pads"] = [0, 0] * dimension
                op.attributes["group"] = w.numel()

                # remove last 2 variable, make conv has exact 3 input
                self.graph.remove_variable(op.inputs[-1])
                self.graph.remove_variable(op.inputs[-1])

                with torch.no_grad():
                    op.inputs[1].value = w
                    op.inputs[2].value = b

    def replace_batchnorm_to_scale(self, dimension: int = 4):
        """Replace Batchnorm to Mul + Add.

        By default this function created a 4d mul + add corresponding to NCHW layout.
        """
        graph = self.graph
        for op in [_ for _ in self.graph.operations.values()]:

            if op.type == "BatchNormalization":
                warning(
                    f"Isolated BatchNormalization({op.name}) was detected, "
                    f"PPQ will replace it to Mul + Add({dimension}D)."
                )

                assert (
                    len(op.parameters) == 4
                ), "BatchNorm should have 4 parameters, namely alpha, beta, mean, var"
                alpha = op.parameters[0].value
                beta = op.parameters[1].value
                mean = op.parameters[2].value
                var = op.parameters[3].value
                epsilon = op.attributes.get("epsilon", 1e-5)

                with torch.no_grad():
                    multiplier = alpha / torch.sqrt(var + epsilon)
                    bias = (-mean) * multiplier + beta

                for var in [_ for _ in op.parameters]:
                    graph.remove_variable(var)
                graph.create_variable(
                    value=multiplier, is_parameter=True, dest_ops=[op]
                )
                op.type = "Mul"
                op.attributes.clear()

                add = graph.create_operation(op_type="Add")
                graph.insert_op_after(op0=add, op1=op)
                graph.create_variable(value=bias, is_parameter=True, dest_ops=[add])

                if dimension > 1:
                    op.parameters[0].value = op.parameters[0].value.reshape(
                        [1, -1] + [1] * (dimension - 2)
                    )
                    add.parameters[0].value = add.parameters[0].value.reshape(
                        [1, -1] + [1] * (dimension - 2)
                    )


class GraphFormatter(GraphCommandProcessor):
    """Graph Formatter offers a bunch of graph editing functions that helps
    modifying your graph.
    """

    @property
    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            GraphCommandType.FORMAT_CLIP,
            GraphCommandType.FORMAT_PAD,
            GraphCommandType.FORMAT_GATHER,
            GraphCommandType.FORMAT_CAST,
            GraphCommandType.FORMAT_INT64_CONSTANT,
            GraphCommandType.DELETE_ISOLATED,
            GraphCommandType.FORMAT_PARAMETERS,
            GraphCommandType.FORMAT_CONSTANT_INPUT,
            GraphCommandType.FORMAT_SLICE,
            GraphCommandType.TRUNCATE_ON_VAR,
            GraphCommandType.FORMAT_RESIZE,
            GraphCommandType.REMOVE_IDENTITY,
            GraphCommandType.CONVERT_TO_TENSOR,
        ]

    def process(self, command: GraphCommand) -> Any:
        if command.command_type == GraphCommandType.FORMAT_CLIP:
            return self.format_clip()
        if command.command_type == GraphCommandType.FORMAT_PAD:
            return self.format_pad()
        if command.command_type == GraphCommandType.FORMAT_GATHER:
            return self.format_gather()
        if command.command_type == GraphCommandType.FORMAT_CAST:
            return self.format_cast()
        if command.command_type == GraphCommandType.DELETE_ISOLATED:
            return self.delete_isolated()
        if command.command_type == GraphCommandType.FORMAT_INT64_CONSTANT:
            return self.format_int64_constant()
        if command.command_type == GraphCommandType.FORMAT_PARAMETERS:
            return self.format_parameter()
        if command.command_type == GraphCommandType.FORMAT_CONSTANT_INPUT:
            return self.remove_constant_input()
        if command.command_type == GraphCommandType.FORMAT_SLICE:
            return self.format_slice()
        if command.command_type == GraphCommandType.FORMAT_RESIZE:
            return self.format_resize()
        if command.command_type == GraphCommandType.TRUNCATE_ON_VAR:
            assert isinstance(
                command, TruncateGraphCommand
            ), "Use TruncateGraphCommand here."
            return self.truncate_on_var(command.var, command.mark_as_output)
        if command.command_type == GraphCommandType.REMOVE_IDENTITY:
            return self.remove_identity()
        if command.command_type == GraphCommandType.CONVERT_TO_TENSOR:
            return self.convert_to_tensor()

    def format_slice(self) -> None:
        """
        Slice: opset1 格式跟其他的不太一样，这个 pass 将 opset1 的 slice 强行转换为 opset 11
        """
        interested_ops = []
        for operation in self.graph.operations.values():
            if operation.type == "Slice":
                if "starts" in operation.attributes:
                    assert (
                        "starts" in operation.attributes
                        and "ends" in operation.attributes
                    ), (
                        f"Invalid Slice Operation Format, Slice operation is expected "
                        "to have axes, starts and ends attributes with opset 1, "
                        f"however your operation {operation.name}, do not have "
                        "completed attributes"
                    )
                    interested_ops.append(operation)

        for slice_op in interested_ops:
            assert isinstance(slice_op, Operation)
            axes = slice_op.attributes.get("axes", None)
            starts = slice_op.attributes["starts"]
            ends = slice_op.attributes["ends"]
            if axes is None:
                axes = [_ for _ in range(starts)]

            slice_op.attributes.pop("starts")
            slice_op.attributes.pop("ends")
            if "axes" in slice_op.attributes:
                slice_op.attributes.pop("axes")
            self.__add_constant_input(slice_op, convert_any_to_tensor(starts))
            self.__add_constant_input(slice_op, convert_any_to_tensor(ends))
            self.__add_constant_input(slice_op, convert_any_to_tensor(axes))

    def format_pad(self) -> None:
        """
        对于不同的模型格式, pad 算子将有两种不同的输入格式：
        for different models, possibly Pad op has the following input formats
            1. pads 参数由第二个输入变量给出
               pads parameter is given by the second input variable
            2. pads 参数被放置于 operation.attribute 中
               pads parameter is set in attribute
        此函数统一 pad 算子行为：所有 pad 算子的 pads 参数均由第二个输入给出
        """
        for op in self.graph.operations.values():
            if op.type == "Pad" and "pads" in op.attributes:
                self.graph.create_variable(
                    value=torch.tensor(op.attributes["pads"]),
                    is_parameter=True,
                    dest_ops=[op],
                )
                op.attributes.clear()

    def format_resize(self) -> None:
        """
        升级 opset 10 的 resize 到 opset 11
        """
        for op in self.graph.operations.values():
            if op.type == "Resize" and len(op.inputs) == 2:
                # 创建一个空的 variable, 这个 variable 没有值与 source_op, 也不是 parameter
                # 这种行为 Onnx 是允许的，这种变量用以传递空值，起到占位符的作用
                self.graph.create_variable(
                    value=None, is_parameter=False, dest_ops=[op]
                )
                op.inputs[1], op.inputs[2] = op.inputs[2], op.inputs[1]

    def format_clip(self) -> None:
        r"""对于不同的模型格式, clip 算子将有两种不同的输入格式：
            1. min, max 参数由 第二、第三个输入变量给出
            2. min, max 参数由 attribute 给出
        此函数统一 clip 算子行为：所有 clip 算子的 min, max 参数第二第三个变量给出
        针对可能存在的 min, max 为空的情况，将其直接置为 2 << 30（保证处理后非空）
        当 min, max 参数由 第二、第三个输入变量给出时，其中一个为空时直接返回 ValueError
        """

        interested_ops = []
        for _, operation in self.graph.operations.items():
            if operation.type == "Clip" and (
                "min" in operation.attributes or "max" in operation.attributes
            ):
                interested_ops.append(operation)
        for op in interested_ops:
            assert isinstance(op, Operation)
            min_value = op.attributes.get("min", -2 << 30)
            max_value = op.attributes.get("max", +2 << 30)
            min_var = Variable(
                name=op.name + "_min", value=min_value, is_parameter=True, dest_ops=[op]
            )
            max_var = Variable(
                name=op.name + "_max", value=max_value, is_parameter=True, dest_ops=[op]
            )
            self.graph.append_variable(min_var)
            self.graph.append_variable(max_var)
            op.inputs.append(min_var)
            op.inputs.append(max_var)
            if "min" in op.attributes:
                op.attributes.pop("min")
            if "max" in op.attributes:
                op.attributes.pop("max")

    def format_gather(self) -> None:
        """gather op 的参数 index 可能由 input variable 给出 但 index
        参数不可以被量化，同时后端运算需要其作为Python 原生类型 因此将其转移到 gather op 的属性上。

        gather op 的参数 axis 可能不存在，此时强制植入 0 作为 axis 参数
        """
        interested_ops = []
        for _, operation in self.graph.operations.items():
            if operation.type == "Gather":
                interested_ops.append(operation)
        for operation in interested_ops:
            assert isinstance(operation, Operation)
            if len(operation.inputs) == 2:
                index_op = operation.inputs[1].source_op
                if index_op is not None and index_op.type == "Constant":
                    index = index_op.attributes["value"]
                    index_value = convert_any_to_python_primary_type(index)
                    self.__delete_constant_input(operation, 1)
                    operation.attributes["gather_index"] = index_value
            if "axis" not in operation.attributes:
                operation.attributes["axis"] = 0

            if "indices" in operation.attributes:
                operation.attributes["gather_index"] = operation.attributes["indices"]
                operation.attributes.pop("indices")

    def format_cast(self) -> None:
        """cast op 的参数 to 默认为 int，使用该函数将其封装为 ppq.core.DataType."""
        interested_ops = []
        for _, operation in self.graph.operations.items():
            assert isinstance(operation, Operation)
            if operation.type == "Cast":
                interested_ops.append(operation)
        for operation in interested_ops:
            assert isinstance(operation, Operation)
            assert "to" in operation.attributes
            operation.attributes["to"] = DataType.from_numpy(operation.attributes["to"])

    def format_int64_constant(self) -> None:
        """convert all int64 constants to int32, check if direct dtype cast is
        available 将所有 int64 的 Constant 转换为 int32 将检查所有 Constant value, 如果 value
        范围在 int32 表示范围内则执行转换。"""
        for operation in self.graph.operations.values():
            if operation.type == "Constant":
                assert "value" in operation.attributes
                value = operation.attributes["value"]

                assert isinstance(value, torch.Tensor)
                if value.dtype != torch.int64:
                    continue

                pvalue = value.tolist()
                check = [0xFFFFFFFF > v >= -0xFFFFFFFF for v in pvalue]

                if all(check):
                    value = value.int()

    def remove_constant_input(self) -> None:
        """部分部署平台不支持 Constant Op 作为算子的输入
            在这种情况下我们使用这个 pass 把它们切换成 Parameter Variable

        Some backend platform doesn't support Constant
        Op, we use this pass to replace it by forcing its value to be a
        parameter variable."""
        removing_ops = []
        for op in self.graph.operations.values():
            if op.type == "Constant":
                if op.num_of_output != 1:
                    error(f"Constant {op.name} has {op.num_of_output} outputs")
                    raise ValueError
                removing_ops.append(op)

        for const_op in removing_ops:
            assert isinstance(const_op, Operation)
            constant_value = const_op.attributes["value"]
            output_var = const_op.outputs[0]
            output_var.is_parameter = True
            output_var.value = constant_value
            self.graph.remove_operation(removing_op=const_op)

    def truncate_on_var(self, var: Variable, mark_as_output: bool):
        """从一个指定位置将图截断.

        Args:
            var (Variable): _description_
            mark_as_output (bool): _description_

        Raises:
            TypeError: _description_
            KeyError: _description_
        """
        graph = self.graph
        if not isinstance(var, Variable):
            error(f"Except Variable, however {type(var)} was given.")
            raise TypeError
        if var.name not in graph.variables:
            error(f"Can not find vairiable {var.name} in current graph")
            raise KeyError

        mark_to_delete, delete_queue, didx = set(), [], 0
        delete_queue.extend(var.dest_ops)
        while didx < len(delete_queue):
            first_op = delete_queue[didx]
            if first_op not in mark_to_delete:
                mark_to_delete.add(first_op)
                delete_queue.extend(graph.get_downstream_operations(first_op))
            didx += 1

        for operation in mark_to_delete:
            graph.remove_operation(operation)

        if mark_as_output:
            graph.mark_variable_as_graph_output(var)

        self.delete_isolated()

    def delete_isolated(self):
        """Remove Isolated Variable from Graph."""
        stop = False
        while not stop:
            blacklist = []
            # delete all operations which are not links to a valid graph output
            for op in self.graph.operations.values():
                if len(self.graph.get_downstream_operations(op)) == 0:
                    output_names = [var.name for var in op.outputs]
                    if all([name not in self.graph.outputs for name in output_names]):
                        blacklist.append(op)

            for op in blacklist:
                for var in op.outputs:
                    self.graph.remove_variable(var)
                self.graph.remove_operation(op)
            stop = len(blacklist) == 0

        stop = False
        while not stop:
            var_blacklist = set()
            # delete all variables that links to invalid operations:
            for var in self.graph.variables.values():
                # 删除无根无输出的变量
                if var.source_op is None and len(var.dest_ops) == 0:
                    var_blacklist.add(var)
                # 删除根节点不在图中的变量
                if (
                    var.source_op is not None
                    and var.source_op.name not in self.graph.operations
                ):
                    var_blacklist.add(var)
                # 删除连接到未知节点的变量
                for op in var.dest_ops:
                    if op.name not in self.graph.operations:
                        var_blacklist.add(var)
                # 删除孤立变量
                if var.source_op is None and var.name not in self.graph.inputs:
                    if len(var.dest_ops) == 0:
                        var_blacklist.add(var)
                # 没有输出的不能删...会影响算子输出顺序...
            for var in var_blacklist:
                self.graph.remove_variable(var)
            stop = len(var_blacklist) == 0

    def format_parameter(self) -> None:
        """Split parameter that has more than 1 dest ops"""
        for var in [_ for _ in self.graph.variables.values()]:
            if var.has_value:
                x = convert_any_to_tensor(var.value)
                var.value = x
            if var.is_parameter and len(var.dest_ops) > 1:
                for op in var.dest_ops:
                    created = self.graph.create_variable(
                        value=var.value.clone(), is_parameter=True
                    )
                    op.inputs[op.inputs.index(var)] = created
                    created.dest_ops.append(op)
                var.dest_ops.clear()
                self.graph.remove_variable(var)

    def __delete_constant_input(self, op: Operation, input_idx: int):
        op_name = op.name
        if op_name not in self._graph.operations:
            error(f"Operation {op_name} not in current graph.")
            raise KeyError
        operation = self._graph.operations[op_name]
        if input_idx >= len(operation.inputs):
            error(
                "Trying to delete an out-of-range input variable, "
                f"has graph been manually changed? Error at Operation {op_name}, "
                f"input_idx: {input_idx}"
            )
            raise ValueError
        input_var = operation.inputs[input_idx]
        source_op = input_var.source_op
        if source_op is None or source_op.type != "Constant":
            error(
                "Trying to delete an non-const input, "
                f"Error at Operation {op_name}, inputs[{input_idx}]"
            )
            raise ValueError
        input_var.dest_ops.pop(input_var.dest_ops.index(operation))
        operation.inputs.pop(input_idx)

        if len(input_var.dest_ops) == 0:
            self.graph.remove_operation(source_op)
            self.graph.remove_variable(input_var)

    def __add_constant_input(self, op: Operation, value: torch.Tensor):
        op_name = op.name
        if op_name not in self._graph.operations:
            error(f"Operation {op_name} not in current graph.")
            raise KeyError
        operation = self._graph.operations[op_name]
        var = Variable(
            name=f"{op_name}_{len(op.inputs) + 1}", value=value, is_parameter=True
        )
        self.graph.append_variable(var)
        var.dest_ops.append(operation)
        operation.inputs.append(var)

    def remove_identity(self):
        """Remove Identity Op from Graph."""
        removing_ops = []
        for op in self.graph.operations.values():
            if op.type == "Identity":
                removing_ops.append(op)

        for op in removing_ops:
            self.graph.remove_operation(op, keep_coherence=True)

    def convert_to_tensor(self):
        """Convert anything inside your network to torch tensor. (Usually from numpy)"""
        for var in self.graph.variables.values():
            if var.has_value:
                var.value = convert_any_to_tensor(var.value)


class GraphMerger(GraphCommandProcessor):
    """Graph Merger implements all graph fusion related functions."""

    @property
    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            # add more extensions in the future
            GraphCommandType.FUSE_BN,
            GraphCommandType.FUSE_BIAS_ADD,
        ]

    def process(self, command: GraphCommand) -> Any:
        if command.command_type == GraphCommandType.FUSE_BN:
            return self.fuse_bn()
        if command.command_type == GraphCommandType.FUSE_BIAS_ADD:
            return self.fuse_bias_add()

    def fuse_bn(self):
        """fuse batchnormalization"""
        search_engine = SearchableGraph(graph=self.graph)
        paths = search_engine.path_matching(
            sp_expr=lambda op: op.type in {"Conv", "Gemm", "ConvTranspose"},
            rp_expr=lambda x, y: False,
            ep_expr=lambda op: op.type == "BatchNormalization",
            direction="down",
        )

        for path in paths:
            path = path.tolist()
            assert len(path) == 2, "Oops seems we got something unexpected."

            computing_op, bn_op = path
            assert isinstance(computing_op, Operation) and isinstance(bn_op, Operation)

            if (
                len(self.graph.get_downstream_operations(computing_op)) != 1
                or len(self.graph.get_upstream_operations(bn_op)) != 1
            ):
                warning(
                    f"PPQ can not merge operation {computing_op.name} and {bn_op.name},"
                    " this is not suppose to happen with your network, network with "
                    "batchnorm inside might not be able to quantize and deploy."
                )
                continue

            assert (
                len(bn_op.parameters) == 4
            ), "BatchNorm should have 4 parameters, namely alpha, beta, mean, var"
            alpha = bn_op.parameters[0].value
            beta = bn_op.parameters[1].value
            mean = bn_op.parameters[2].value
            var = bn_op.parameters[3].value
            epsilon = bn_op.attributes.get("epsilon", 1e-5)

            if computing_op.num_of_parameter == 1:
                w = computing_op.parameters[0].value  # no bias.
                assert isinstance(
                    w, torch.Tensor
                ), "values of parameters are assumed as torch Tensor"
                if computing_op.type == "ConvTranspose":
                    b = torch.zeros(
                        w.shape[1] * computing_op.attributes.get("group", 1)
                    )
                elif (
                    computing_op.type == "Gemm"
                    and computing_op.attributes.get("transB", 0) == 0
                ):
                    b = torch.zeros(w.shape[1])
                else:
                    b = torch.zeros(w.shape[0])
            else:
                w, b = [var.value for var in computing_op.parameters[:2]]  # has bias.

            if computing_op.type == "Conv":

                # calculate new weight and bias
                scale = alpha / torch.sqrt(var + epsilon)
                w = w * scale.reshape([-1] + [1] * (w.ndim - 1))
                b = alpha * (b - mean) / torch.sqrt(var + epsilon) + beta

            elif computing_op.type == "Gemm":

                # calculate new weight and bias
                scale = alpha / torch.sqrt(var + epsilon)
                if computing_op.attributes.get("transB", 0):
                    w = w * scale.reshape([-1, 1])
                else:
                    w = w * scale.reshape([1, -1])
                b = alpha * (b - mean) / torch.sqrt(var + epsilon) + beta

            elif computing_op.type == "ConvTranspose":

                scale = alpha / torch.sqrt(var + epsilon)
                group = computing_op.attributes.get("group", 1)
                scale = scale.reshape([group, 1, -1] + [1] * (w.ndim - 2))
                w = w.reshape([group, -1] + list(w.shape[1:])) * scale
                w = w.reshape([w.shape[0] * w.shape[1]] + list(w.shape[2:]))
                b = alpha * (b - mean) / torch.sqrt(var + epsilon) + beta
            else:
                raise TypeError(
                    f"Unexpected op type {computing_op.type}. "
                    f"Can not merge {computing_op.name} with {bn_op.name}"
                )

            # create new op and variable
            merged_op = Operation(
                computing_op.name,
                op_type=computing_op.type,
                attributes=computing_op.attributes.copy(),
            )
            weight_var = Variable(computing_op.name + "_weight", w, True, [merged_op])
            bias_var = Variable(computing_op.name + "_bias", b, True, [merged_op])

            # replace & dirty work
            input_var = computing_op.inputs[0]
            output_var = bn_op.outputs[0]

            input_var.dest_ops.remove(computing_op)
            input_var.dest_ops.append(merged_op)

            output_var.source_op = merged_op

            # delete old operations
            computing_op.inputs.pop(0)
            bn_op.outputs.clear()
            self.graph.remove_operation(computing_op)
            self.graph.remove_operation(bn_op)

            # insert new
            self.graph.append_operation(merged_op)
            merged_op.inputs.extend([input_var, weight_var, bias_var])
            merged_op.outputs.extend([output_var])

            self.graph.append_variable(weight_var)
            self.graph.append_variable(bias_var)

    def fuse_gemm(self):
        """Fuse MatMul + add into a signal Gemm
            Single Matmul will be replaced with Gemm

        Returns:
            _type_: _description_
        """

        def _is_replaceable(op: Operation) -> bool:
            return op.inputs[0].is_parameter or op.inputs[1].is_parameter

        search_engine = SearchableGraph(graph=self.graph)
        patterns = search_engine.pattern_matching(
            patterns=["MatMul", "Add"], edges=[[0, 1]], exclusive=True
        )
        for pattern in patterns:
            matmul, add = pattern

            if not _is_replaceable(matmul):
                continue

            matmul.type = "Gemm"

            matmul_out = matmul.outputs[0]
            add_out = add.outputs[0]

            if matmul.inputs[0].is_parameter:
                temp = matmul.inputs[0]
                matmul.inputs[0] = matmul.inputs[1]
                matmul.inputs[1] = temp

            assert len(add.inputs) == 2, "Oops, seems we got some problem here."
            var1, var2 = add.inputs
            bias_var = None

            if var1.source_op == matmul and var2.is_parameter:
                bias_var = var2

            if var2.source_op == matmul and var1.is_parameter:
                bias_var = var1

            # can not find a valid bias, just skip add.
            if bias_var is None:
                continue

            if len(bias_var.value.shape) == 1:
                if bias_var.value.shape[0] == matmul.parameters[0].value.shape[-1]:
                    matmul.attributes["transB"] = 1
                    weight_val = matmul.parameters[0].value

                    matmul.parameters[0].value = weight_val.transpose(-1, -2)

                    bias_var.dest_ops.clear()
                    add.inputs.remove(bias_var)

                    # remove bias add, move bias to matmul
                    self.graph.remove_operation(add)
                    self.graph.create_link_with_op(None, matmul, bias_var)
                    self.graph.create_link_with_var(matmul_out, add_out)
                elif bias_var.value.shape[0] == matmul.parameters[0].value.shape[-2]:

                    bias_var.dest_ops.clear()
                    add.inputs.remove(bias_var)
                    # remove bias add, move bias to matmul
                    self.graph.remove_operation(add)
                    self.graph.create_link_with_op(None, matmul, bias_var)
                    self.graph.create_link_with_var(matmul_out, add_out)

        # process single gemm
        for op in self.graph.operations.values():
            if op.type == "MatMul":
                if not _is_replaceable(op):
                    continue
                op.type = "Gemm"

    def fuse_layernorm(self, exclusive_search: bool = False):  # noqa: C901
        """Fuse Layernormalization with pattern matching."""

        def _fuse(
            rm1: Operation,
            rm2: Operation,
            eps: Operation,
            scale: torch.Tensor,
            bias: Optional[torch.Tensor],
            layernorm_input_var: Variable,
            layernorm_output_var: Variable,
        ) -> None | Operation:

            if rm2.type == rm1.type == "ReduceMean":
                if "axes" not in rm1.attributes:
                    return None
                if "axes" not in rm2.attributes:
                    return None
                if rm1.attributes["axes"] != rm2.attributes["axes"]:
                    return None
                layernorm_axis = rm1.attributes["axes"]
                if isinstance(layernorm_axis, list):
                    if len(layernorm_axis) != 1:
                        return None
                    layernorm_axis = layernorm_axis[0]
                if not isinstance(layernorm_axis, int):
                    return None
            else:
                layernorm_axis = -1

            if not eps.inputs[-1].is_parameter:
                return None
            value = eps.inputs[-1].value
            value = convert_any_to_tensor(value).cpu()
            if value.numel() != 1:
                return None
            layernorm_eps = value.item()
            source_op = layernorm_output_var.source_op
            if source_op is not None:
                source_op.outputs.clear()
            layernorm = self.graph.create_operation(
                op_type="LayerNormalization",
                attributes={
                    "axis": layernorm_axis,
                    "epsilon": layernorm_eps,
                    "stash_type": 0,
                },
                inputs=[
                    layernorm_input_var,
                    self.graph.create_variable(value=scale, is_parameter=True),
                ],
                outputs=[layernorm_output_var],
            )

            if bias is not None:
                self.graph.create_link_with_op(
                    None,
                    layernorm,
                    self.graph.create_variable(value=bias, is_parameter=True),
                )
            return layernorm

        search_engine = SearchableGraph(graph=self.graph)
        fused = False

        # pattern 1:
        # y = Sub(x, ReduceMean(x))
        # Add(Mul(Div(Sqrt(Add(ReduceMean(Pow(y)))), y)))
        matches = search_engine.pattern_matching(
            patterns=[
                lambda op: True,
                lambda op: op.type in {"ReduceMean", "GlobalAveragePool"},
                "Sub",
                "Pow",
                lambda op: op.type in {"ReduceMean", "GlobalAveragePool"},
                "Add",
                "Sqrt",
                "Div",
                "Mul",
            ],
            edges=[
                [0, 1],
                [0, 2],
                [1, 2],
                [2, 3],
                [2, 7],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
            ],
            exclusive=exclusive_search,
        )

        for _, rm1, sub, pow, rm2, add, sqrt, div, mul in matches:
            layernorm_ops = [rm1, sub, pow, rm2, add, sqrt, div, mul]

            layernorm_scale = mul.inputs[-1].value
            layernorm_output_var = div.outputs[0]
            layernorm_input_var = sub.inputs[0]

            # bias check
            layernorm_bias = None
            next_op = self.graph.get_downstream_operations(mul)
            if len(next_op) == 1 and (next_op[0].type == "Add"):
                bias_op = next_op[0]
                if bias_op.inputs[-1].is_parameter:
                    layernorm_bias = bias_op.inputs[-1].value
                    layernorm_output_var = bias_op.outputs[0]
                    layernorm_ops.append(bias_op)

            layernorm = _fuse(
                rm1=rm1,
                rm2=rm2,
                eps=add,
                scale=layernorm_scale,
                bias=layernorm_bias,
                layernorm_input_var=layernorm_input_var,
                layernorm_output_var=layernorm_output_var,
            )

            if layernorm is not None:
                # delete merged ops
                for op in layernorm_ops:
                    assert isinstance(op, Operation)
                    for var in op.inputs + op.outputs:
                        if var != layernorm_input_var and var != layernorm_output_var:
                            self.graph.remove_variable(var)
                    self.graph.remove_operation(op)
                    fused = True

        # pattern 2:
        matches = search_engine.pattern_matching(
            patterns=[
                lambda op: True,
                lambda op: op.type in {"ReduceMean", "GlobalAveragePool"},
                "Sub",
                "Mul",
                lambda op: op.type in {"ReduceMean", "GlobalAveragePool"},
                "Add",
                "Sqrt",
                "Reciprocal",
                "Mul",
                "Mul",
                "Sub",
                "Mul",
                "Add",
            ],
            edges=[
                [0, 1],
                [0, 2],
                [0, 11],
                [1, 2],
                [1, 9],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [8, 11],
                [9, 10],
                [11, 12],
                [10, 12],
            ],
            exclusive=exclusive_search,
        )

        for (
            _,
            rm1,
            sub1,
            mul,
            rm2,
            add1,
            sqrt,
            recipro,
            mul2,
            mul3,
            sub2,
            mul4,
            add2,
        ) in matches:
            layernorm_ops = [
                rm1,
                sub1,
                mul,
                rm2,
                add1,
                sqrt,
                recipro,
                mul2,
                mul3,
                sub2,
                mul4,
                add2,
            ]

            # mul check
            if not mul2.inputs[-1].is_parameter:
                continue
            layernorm_scale = mul2.inputs[-1].value
            layernorm_output_var = add2.outputs[0]
            layernorm_input_var = sub1.inputs[0]
            layernorm_bias = sub2.inputs[0].value

            layernorm = _fuse(
                rm1=rm1,
                rm2=rm2,
                eps=add1,
                scale=layernorm_scale,
                bias=layernorm_bias,
                layernorm_input_var=layernorm_input_var,
                layernorm_output_var=layernorm_output_var,
            )

            if layernorm is not None:
                # delete merged ops
                for op in layernorm_ops:
                    assert isinstance(op, Operation)
                    for var in op.inputs + op.outputs:
                        if var != layernorm_input_var and var != layernorm_output_var:
                            self.graph.remove_variable(var)
                    self.graph.remove_operation(op)
                fused = True

        # final check, if no valid pattern was found, we give a warning.
        if not fused:
            warning("No valid layernorm pattern was found, check your graph again.")

    def fuse_skiplayernorm(self):
        """Fuse Add + Layernorm to SkipLayernorm, SkipLayernorm is a plugin operation
        defined by TensorRT
        """
        fused = False
        search_engine = SearchableGraph(graph=self.graph)

        matches = search_engine.pattern_matching(
            patterns=["Add", "LayerNormalization"], edges=[[0, 1]], exclusive=True
        )
        for add, layernorm in matches:
            # Skip connection can not be constant.
            if add.num_of_parameter == 0:
                input_vars = add.inputs.copy()
                output_var = add.outputs[0]
                self.graph.remove_operation(add)
                self.graph.remove_variable(output_var)

                for var in input_vars:
                    var.dest_ops.append(layernorm)
                    layernorm.inputs.append(var)

                layernorm.type = "skipLayerNormPlugin"
                fused = True
        # final check, if no valid pattern was found, we give a warning.
        if not fused:
            warning(
                "No valid Skip Layernorm pattern was found, check your graph again."
            )

    def fuse_gelu(self):
        """Fuse Gelu

        Pattern: * - Div - Erf - Add - Mul - Mul
                   |                 |
                   -------------------
        """
        fused = False
        search_engine = SearchableGraph(graph=self.graph)

        matches = search_engine.pattern_matching(
            patterns=[lambda op: True, "Div", "Erf", "Add", "Mul", "Mul"],
            edges=[[0, 1], [1, 2], [2, 3], [3, 4], [0, 4], [4, 5]],
            exclusive=True,
        )

        for _, div, erf, add, mul1, mul2 in matches:
            removing_var = []
            removing_var.extend(div.outputs)
            removing_var.extend(erf.outputs)
            removing_var.extend(add.outputs)
            removing_var.extend(mul1.outputs)

            self.graph.remove_operation(div)
            self.graph.remove_operation(erf)
            self.graph.remove_operation(add)
            self.graph.remove_operation(mul1)
            for var in removing_var:
                self.graph.remove_variable(var)

            input_vars = _.outputs.copy()
            output_vars = mul2.outputs.copy()

            self.graph.remove_operation(mul2)
            self.graph.create_operation(
                op_type="Gelu", inputs=input_vars, outputs=output_vars
            )
            assert len(input_vars) == 1, "Fusion failed, Pattern unrecognized."
            fused = True

        # final check, if no valid pattern was found, we give a warning.
        if not fused:
            warning("No valid Gelu pattern was found, check your graph again.")

    def fuse_bias_add(self):
        """
        Fuse Pattern like Conv + Add, ConvTranspose + Add, Gemm + Add
        This fusion will require a constant input as bias.
        """
        graph = self.graph
        for op in [_ for _ in graph.operations.values()]:
            if op.type in {"Conv", "ConvTranspose", "Gemm"}:
                # check if current op has only 1 downstream op
                channel_dimension = 1  # NCHW, NCHWD, NCH
                if op.type == "Gemm":
                    channel_dimension = 0
                if len(graph.get_downstream_operations(op)) == 1:
                    down = graph.get_downstream_operations(op)[0]

                    if down.type == "Add":
                        if down.num_of_parameter != 1:
                            continue
                        if op.num_of_input == 3:  # already has a bias
                            continue

                        bias = down.parameters[0]
                        if op.type not in {"Gemm"}:
                            # check if it is a bias add
                            if not bias.value.dim() == op.parameters[0].value.dim():
                                continue
                            if not bias.value.squeeze().dim() <= 1:
                                continue
                            if bias.value.shape[channel_dimension] == 1:
                                bias.value = bias.value.reshape(
                                    [-1]
                                )  # fuse add even if oc=1
                            else:
                                bias.value = (
                                    bias.value.squeeze()
                                )  # conv bias can only be 1d
                        else:
                            # Gemm bias can be any shape.
                            pass

                        # ready for fusion
                        graph.create_variable(
                            is_parameter=True, value=bias.value, dest_ops=[op]
                        )
                        graph.remove_operation(removing_op=down, keep_coherence=True)

    def fuse_selfattention(self):  # noqa: C901
        search_engine = SearchableGraph(graph=self.graph)
        matches = search_engine.pattern_matching(
            patterns=["MatMul", "Add", "Softmax", "MatMul"],
            edges=[[0, 1], [1, 2], [2, 3]],
            exclusive=False,
        )

        for m1, add, softmax, m2 in matches:
            trans_q, trans_k, trans_v, trans_o = None, None, None, None
            perm_q, perm_k, perm_v, perm_o = None, None, None, None

            # check pattern
            if m1.num_of_parameter != 0 or m2.num_of_parameter != 0:
                continue
            if m2.inputs[0].source_op != softmax:
                continue

            # source op of Q
            sq = m1.inputs[0].source_op
            if sq is not None and sq.type == "Transpose":
                trans_q = sq
                perm_q = trans_q.attributes["perm"]

            # source op of K
            sk = m1.inputs[1].source_op
            if sk is not None and sk.type == "Transpose":
                trans_k = sk
                perm_k = trans_k.attributes["perm"]

            # source op of V
            sv = m2.inputs[1].source_op
            if sv is not None and sv.type == "Transpose":
                trans_v = sv
                perm_v = trans_v.attributes["perm"]

            # output op of O
            oo = m2.outputs[0].dest_ops
            if len(oo) == 1 and oo[0].type == "Transpose":
                trans_o = oo[0]
                perm_o = trans_o.attributes["perm"]

            for op in [trans_q, trans_k, trans_v, trans_o, softmax]:
                if op is not None:
                    self.graph.remove_operation(op, keep_coherence=True)
            for var in [add.outputs[0], m1.outputs[0]]:
                self.graph.remove_variable(var)

            q_var, k_var, v_var = m1.inputs[0], m1.inputs[1], m2.inputs[1]
            mask_var, o_var = add.inputs[0], m2.outputs[0]

            for op in m1, add, m2:
                self.graph.remove_operation(op)

            op = self.graph.create_operation(
                op_type="SelfAttention",
                attributes={
                    "TransQ": perm_q,
                    "TransK": perm_k,
                    "TransV": perm_v,
                    "TransO": perm_o,
                },
                inputs=[q_var, k_var, v_var, mask_var],
                outputs=[o_var],
            )

            # remove empty transpose
            non_empty_attr = {}
            for k, v in op.attributes.values():
                if v is not None:
                    non_empty_attr[k] = v
            op.attributes = non_empty_attr

        matches = search_engine.pattern_matching(
            patterns=["MatMul", "Softmax", "MatMul"],
            edges=[[0, 1], [1, 2]],
            exclusive=False,
        )

        # self-attention with no mask
        for m1, softmax, m2 in matches:
            trans_q, trans_k, trans_v, trans_o = None, None, None, None
            perm_q, perm_k, perm_v, perm_o = None, None, None, None

            # check pattern
            if m1.num_of_parameter != 0 or m2.num_of_parameter != 0:
                continue
            if m2.inputs[0].source_op != softmax:
                continue

            # source op of Q
            sq = m1.inputs[0].source_op
            if sq is not None and sq.type == "Transpose":
                trans_q = sq
                perm_q = trans_q.attributes["perm"]

            # source op of K
            sk = m1.inputs[1].source_op
            if sk is not None and sk.type == "Transpose":
                trans_k = sk
                perm_k = trans_k.attributes["perm"]

            # source op of V
            sv = m2.inputs[1].source_op
            if sv is not None and sv.type == "Transpose":
                trans_v = sv
                perm_v = trans_v.attributes["perm"]

            # output op of O
            oo = m2.outputs[0].dest_ops
            if len(oo) == 1 and oo[0].type == "Transpose":
                trans_o = oo[0]
                perm_o = trans_o.attributes["perm"]

            for op in [trans_q, trans_k, trans_v, trans_o, softmax]:
                if op is not None:
                    self.graph.remove_operation(op, keep_coherence=True)
            for var in [m1.outputs[0]]:
                self.graph.remove_variable(var)

            q_var, k_var, v_var = m1.inputs[0], m1.inputs[1], m2.inputs[1]
            o_var = m2.outputs[0]

            for op in m1, m2:
                self.graph.remove_operation(op)

            op = self.graph.create_operation(
                op_type="SelfAttention",
                attributes={
                    "TransQ": perm_q,
                    "TransK": perm_k,
                    "TransV": perm_v,
                    "TransO": perm_o,
                },
                inputs=[q_var, k_var, v_var, self.graph.create_variable()],
                outputs=[o_var],
            )

            # remove empty transpose
            non_empty_attr = {}
            for k, v in op.attributes.values():
                if v is not None:
                    non_empty_attr[k] = v
            op.attributes = non_empty_attr
