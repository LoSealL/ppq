import json
import os
from abc import abstractmethod
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch

from mppq.data import convert_any_to_python_primary_type as _convert
from mppq.defs import SingletonMeta
from mppq.ir.base.opdef import Operation, Variable
from mppq.ir.base.quantize import QuantableOperation
from mppq.logger import error, warning
from mppq.quant import QuantizationStates, TargetPrecision
from mppq.storage import Serializable


class BaseGraph(Serializable):
    r"""Graph is a PPQ Internal Represtation Data Structure.

    计算图是一个有向图，其中节点对应于操作或变量。
    变量可以将其值输入到操作中，操作可以将其输出输入到其他操作中。这样，图中的每个节点都定义了变量的函数。

    输入节点和从节点输出的值称为张量，这只是一个多维数组的花哨说法。
    因此，它也包括标量、向量和矩阵以及更高秩的张量。

    PPQ 创建的计算图包含量化信息以及操作和变量。因此可以说它是一个专为量化设计的计算图。

    所有的量化相关信息都存储在图及其操作中。
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.operations: Dict[str, Operation] = {}
        self.variables: Dict[str, Variable] = {}
        self.inputs: Dict[str, Variable] = {}
        self.outputs: Dict[str, Variable] = {}
        self.name = name
        self._num_of_generated_var = 0
        self._num_of_generated_op = 0
        self._detail: Dict[str, Any] = {}

    def parameters(self) -> List[torch.Tensor]:
        r"""获取所有参数变量"""
        parameters = []
        for var in self.variables.values():
            if var.is_parameter:
                parameters.append(var.value)
        return parameters

    def set_extension_attrib(self, attrib: str, value: Any):
        r"""设置扩展属性"""
        self._detail[attrib] = value

    @property
    def extension_attrib(self):
        r"""获取扩展属性"""
        return self._detail

    def append_operation(self, operation: Operation):
        r"""添加一个新算子到图中"""
        assert isinstance(operation, Operation)
        for var in operation.inputs + operation.outputs:
            if var.name not in self.variables:
                self.append_variable(var)
        if operation.name in self.operations:
            raise KeyError(f"Duplicated Operation({operation}) was found!")
        self.operations[operation.name] = operation

    def append_variable(self, var: Variable):
        r"""手动添加一个变量到图中。

        Note:

            变量的消费者必须提前添加或已存在。
        """
        assert isinstance(var, Variable)
        if not all([dest_op.name in self.operations for dest_op in var.dest_ops]):
            raise RuntimeError(
                f"Inserting Variable {var} has a related Operation(dest_op) "
                "which are not included in this graph yet, "
                "insert such Operations before inserting this."
            )
        if var.name in self.variables:
            raise KeyError(f"Duplicated Variable({var}) was found!")
        self.variables[var.name] = var

    def get_downstream_operations(self, operation: Operation) -> List[Operation]:
        r"""获取算子的后驱算子"""
        assert isinstance(operation, Operation)
        if operation.name not in self.operations:
            raise KeyError(f"Operation {operation.name} not in current graph.")
        downstream_ops = []
        for output_var in operation.outputs:
            downstream_ops.extend(output_var.dest_ops)
        return downstream_ops

    def get_upstream_operations(self, operation: Operation) -> List[Operation]:
        r"""获取算子的前驱算子"""
        assert isinstance(operation, Operation)
        if operation.name not in self.operations:
            raise KeyError(f"Operation {operation.name} not in current graph.")
        upstream_ops = []
        for input_var in operation.inputs:
            if input_var.source_op is not None:
                upstream_ops.append(input_var.source_op)
        return upstream_ops

    def topological_sort(self) -> List[Operation]:
        r"""返回拓扑排序后的算子列表"""
        visited = {operation.name: False for operation in self.operations.values()}
        sort_ret, pop_list = [], deque()
        num_of_inputs = {
            operation.name: len(self.get_upstream_operations(operation))
            for operation in self.operations.values()
        }

        # initialization
        for op_name, n_input in num_of_inputs.items():
            if n_input == 0:
                pop_list.append(op_name)

        # topological sort
        for _ in range(len(visited)):
            if len(pop_list) == 0:
                break
            op_name = pop_list.popleft()
            op = self.operations[op_name]
            for post_op in self.get_downstream_operations(op):
                num_of_inputs[post_op.name] -= 1
                if num_of_inputs[post_op.name] == 0:
                    pop_list.append(post_op.name)
            visited[op.name] = True
            sort_ret.append(op)
        if all(visited.values()):
            return sort_ret
        else:
            error("Some operation can not be sorted (might due to circular reference)")
            error("\n".join(op_name for op_name in visited if not visited[op_name]))
            raise RuntimeError("Topological Sort failed.")

    def insert_op_on_var(self, inserting_op: Operation, var: str):
        """Insert one operation to current graph. Inserting operation will
        replace var.dest_ops and automatically connect to inserting_op.

        Before insertion:
            op1 -> var -> op2

        After insertion:
            op1 -> var -> inserting_op -> link_var(generated) -> op2

        ATTENTION:

            Inserting operation must be an empty operation with no input and output
            variables linked to it.
        """
        if not isinstance(var, str):
            raise TypeError(
                f"Needs a variable name(str) here, however {type(var)} was given"
            )
        if var not in self.variables:
            raise KeyError(
                f"Can not inserting operation at variable {var}, variable not found."
            )
        if inserting_op.num_of_input != 0 or inserting_op.num_of_output != 0:
            raise PermissionError(
                "Can only insert operation with no input and output variables."
            )

        variable = self.variables[var]
        # add to graph.
        if inserting_op.name not in self.operations:
            self.append_operation(inserting_op)
        # create all links.
        link_var = self.create_variable(
            is_parameter=False,
            dest_ops=variable.dest_ops.copy(),
            source_op=inserting_op,
        )

        inserting_op.inputs.append(variable)
        inserting_op.outputs.append(link_var)

        variable.dest_ops.clear()
        variable.dest_ops.append(inserting_op)

        for op in link_var.dest_ops:
            op.inputs[op.inputs.index(variable)] = link_var

        if var in self.outputs:
            self.outputs.pop(var)
            self.outputs[link_var.name] = link_var

    def insert_op_before(self, op0: Operation, op1: Operation, input_idx: int = 0):
        """Insert an op just before given op. This function will insert given
        op A to variable B.inputs[input_idx]

        Args:
            op0 (Operation): Inserting Op, should has no input and output variable that
                links to it.
            op1 (Operation): before this op.
            input_idx (int, optional): For case that B has more than 1 input variable,
                user should use parameter input_idx to identify which variable is used.
        """
        if input_idx >= op1.num_of_input:
            raise ValueError("Input index out of range.")
        if op0.num_of_input != 0 or op0.num_of_output != 0:
            raise ValueError(
                "Can only insert op that has no input and output variable."
            )
        var = op1.inputs[input_idx]
        var.dest_ops[var.dest_ops.index(op1)] = op0
        op1.inputs[input_idx] = self.create_variable()
        op1.inputs[input_idx].source_op = op0
        op1.inputs[input_idx].dest_ops.append(op1)
        op0.inputs.append(var)
        op0.outputs.append(op1.inputs[input_idx])

    def insert_op_after(self, op0: Operation, op1: Operation, output_idx: int = 0):
        """Insert an op just after given op. This function will insert given op
        A to variable B.outputs[output_idx]

        Args:
            op0 (Operation): Inserting Op, should has no input and output variable that
                links to it.
            op1 (Operation): after this op.
            output_idx (int, optional): For case that B has more than 1 output variable,
                user should use parameter output_idx to identify which variable is used.
        """
        if output_idx >= op1.num_of_output:
            raise ValueError("Output index out of range.")
        if op0.num_of_input != 0 or op0.num_of_output != 0:
            raise ValueError(
                "Can only insert op that has no input and output variable."
            )
        var = op1.outputs[output_idx]
        var.source_op = op0
        op1.outputs[output_idx] = self.create_variable()
        op1.outputs[output_idx].source_op = op1
        op1.outputs[output_idx].dest_ops.append(op0)
        op0.outputs.append(var)
        op0.inputs.append(op1.outputs[output_idx])

    def create_link_with_op(
        self,
        op0: Optional[Operation],
        op1: Optional[Operation],
        variable: Optional[Variable] = None,
    ):
        """Create a link with given variable from upstream_op to downstream_op
        variable will be appended to upstream_op's output and downstream_op's
        input given variable must have empty source_op or its source_op == upstream_op.

        Sometime you may want to link a single upstream_op to many downstream_ops with
        a same variable, you are supposed to invoke this function for each downstream_op
        then.

        You can set upstream_op = None if your variable is a parameter variable.

        Example::

            create_link_with_op(var1, op1, op2)
            create_link_with_op(var1, op1, op3)

        Will makes:
                  --> op2
            op1 --|
                  --> op3
        """
        if variable is None:
            variable = self.create_variable()
        if variable.name not in self.variables:
            raise KeyError(
                f"Can not find your variable {variable.name} in current graph."
            )
        if op0 is not None and op0.name not in self.operations:
            raise KeyError(f"Can not find your operation {op0.name} in current graph.")
        if op1 is not None and op1.name not in self.operations:
            raise KeyError(f"Can not find your operation {op1.name} in current graph.")

        if variable.source_op is None and op0 is not None:
            variable.source_op = op0
        if variable.source_op != op0:
            raise PermissionError(
                f"Can not create link with variable {variable}, "
                f"cause its source operations != {op0}"
            )

        # For complex graph, following logic might have some error.
        if op0 is not None and variable not in op0.outputs:
            op0.outputs.append(variable)
        if op1 is None:
            return
        if op1 is not None and variable not in op1.inputs:
            variable.dest_ops.append(op1)
            op1.inputs.append(variable)
        else:
            variable.dest_ops.append(op1)
            op1.inputs.append(variable)
            warning(
                "You are trying to link variable with operation, "
                f"however Variable {variable.name} has already linked "
                f"with downstream op {op1.name}"
            )

    def create_link_with_var(self, op0: Variable, op1: Variable):
        """connect upstream_variable.source_op with downstream_variable.dest_ops,
        downstream variable will be eliminated by this function.

        downstream_variable must have None as its source_op.
        """
        if op0 is not None and op0.name not in self.variables:
            raise KeyError(f"Can not find your variable {op0.name} in current graph.")
        if op1 is not None and op1.name not in self.variables:
            raise KeyError(f"Can not find your variable {op1.name} in current graph.")

        if op1.source_op is not None:
            raise PermissionError(
                f"Can not create link with variable {op0.name} & {op1.name}, "
                "Cause downstream variable has a non-empty source op"
            )

        dest_ops = op1.dest_ops
        for dest_op in dest_ops:
            dest_op.inputs[dest_op.inputs.index(op1)] = op0
            op0.dest_ops.append(dest_op)
        op1.dest_ops.clear()
        self.remove_variable(op1)
        return self

    def remove_operation(
        self,
        removing_op: Operation,
        keep_coherence: bool = False,
        remove_unlinked_variable: bool = False,
    ):
        """Remove operation from graph, this function will unlink removing
        operation from current graph, pop it from graph.operations, and remove
        it from all its input and output variables.

        Parameters of this removing operations will be removed from graph by this
        function, without warning.

        Args:
            keep_coherence (bool): if keep_coherence = True, PPQ will link downstream
                operations of removing op to the upstream operation. If there is more
                than 1 input and output variable, PPQ will link input[0] with output[0]
        """
        if removing_op.name not in self.operations:
            raise KeyError(
                f"Can not remove operation {removing_op.name}, operation not found."
            )

        # removing all parameters first.
        for parameter in removing_op.inputs.copy():
            if keep_coherence and removing_op.type in {"Constant", "Identity"}:
                break
            if parameter.is_parameter:
                parameter.dest_ops.clear()
                # pylint: disable=protected-access
                parameter._value = None  # clear memory.
                removing_op.inputs.remove(parameter)

                self.variables.pop(parameter.name)

        related_vars = [var for var in removing_op.inputs + removing_op.outputs]
        input_var, output_var = (
            removing_op.inputs[0] if removing_op.num_of_input >= 1 else None,
            removing_op.outputs[0] if removing_op.num_of_output >= 1 else None,
        )

        # remove operation from its output variables
        for _output_var in removing_op.outputs:
            _output_var.source_op = None
        removing_op.outputs.clear()

        # remove operation from its input variables
        for _input_var in removing_op.inputs:
            if removing_op in _input_var.dest_ops:
                _input_var.dest_ops.remove(removing_op)
        removing_op.inputs.clear()

        if input_var is not None and output_var is not None and keep_coherence:

            removing_var = output_var
            dest_ops = removing_var.dest_ops
            is_graph_output = removing_var.name in self.outputs

            for op in dest_ops:
                op.inputs[op.inputs.index(removing_var)] = input_var
                input_var.dest_ops.append(op)
            removing_var.dest_ops.clear()
            removing_var.source_op = None
            self.remove_variable(removing_var)

            if is_graph_output:
                self.mark_variable_as_graph_output(input_var)

        self.operations.pop(removing_op.name)

        if remove_unlinked_variable:
            for var in related_vars:
                if (
                    var.source_op is None
                    and len(var.dest_ops) == 0
                    and var.name in self.variables
                ):
                    self.remove_variable(var)

        return self

    def remove_variable(self, removing_var: Variable):
        """Remove variable from graph, this function will unlink removing
        variable from current graph, pop it from graph.variables, and remove it
        from its source op and dest ops.
        """
        if removing_var.name not in self.variables:
            raise KeyError(
                f"Can not remove variable {removing_var.name}, variable not found."
            )

        # remove from source operation
        source_op = removing_var.source_op
        if source_op is not None:
            assert isinstance(source_op, Operation)
            if removing_var in source_op.outputs:
                source_op.outputs.remove(removing_var)
            removing_var.source_op = None

        # remove from all dest ops
        for dest_op in removing_var.dest_ops:
            assert isinstance(dest_op, Operation)
            if removing_var in dest_op.inputs:
                dest_op.inputs.remove(removing_var)
        removing_var.dest_ops.clear()

        if removing_var.name in self.outputs:
            self.outputs.pop(removing_var.name)

        if removing_var.name in self.inputs:
            self.inputs.pop(removing_var.name)

        self.variables.pop(removing_var.name)
        return self

    def create_operation(
        self,
        op_type: str,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        platform: TargetPrecision = TargetPrecision.UNSPECIFIED,
        inputs: Optional[List[Variable]] = None,
        outputs: Optional[List[Variable]] = None,
    ) -> Operation:
        """Create an operation and attach it it current graph. op_type is
        mandatory here, however op_name is not required. PPQ will automatically
        generates a name for your operation:
        PPQ_Operation_{self._num_of_generated_op}.

        Use this function carefully, cause once your network is quantized, simply
        create an operation via this function might cause unexpected error.
        Beawre that operation created by this function has no meta data and
        quantization info, which is needed to export and executing your graph.

        Do not set inputs and outputs via this function, to link your operation with
        others, use graph.create_link_with_var instead.
        """
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []

        if name is None:
            name = f"PPQ_Operation_{self._num_of_generated_op}"
            self._num_of_generated_op += 1

        if not isinstance(inputs, list):
            raise TypeError(
                f"A list of input variable is required for creating operation, "
                f"however {type(inputs)} was given"
            )
        created = Operation(
            name=name,
            op_type=op_type,
            attributes=attributes or {},
            precision=platform,
            inputs=inputs,
            outputs=outputs,
        )
        self.append_operation(created)

        for item in inputs:
            if not isinstance(item, Variable):
                raise TypeError(
                    f"A list contains variables is required for creating operation, "
                    f"however there is a {type(item)} in your input list."
                )
            item.dest_ops.append(created)

        if not isinstance(outputs, list):
            raise TypeError(
                f"A list of output variable is required for creating operation, "
                f"however {type(inputs)} was given"
            )
        for item in outputs:
            if not isinstance(item, Variable):
                raise TypeError(
                    f"A list contains variables is required for creating operation, "
                    f"however there is a {type(item)} in your output list."
                )
            item.source_op = created
        return created

    def create_variable(
        self,
        name: Optional[str] = None,
        value: Optional[Any] = None,
        is_parameter: bool = False,
        dest_ops: Optional[List[Operation]] = None,
        source_op: Optional[Operation] = None,
    ) -> Variable:
        """Create a variable and attach it it current graph. PPQ will
        automatically generates a name for your variable:
        PPQ_Variable_{self._num_of_generated_op}.

        Use this function carefully, cause once your network is quantized, simply
        create an variable via this function might cause unexpected error.
        You'd better invoke this function before running your quantizer.

        If dest_ops and source_op is not None, this function will auto link created
        variable with them.
        """
        if name is None:
            name = f"PPQ_Variable_{self._num_of_generated_var}"
            self._num_of_generated_var += 1

        created = Variable(
            name=name,
            value=value,
            is_parameter=is_parameter,
            dest_ops=dest_ops,
            source_op=source_op,
        )

        if dest_ops is not None:
            for op in dest_ops:
                if not isinstance(op, Operation):
                    raise TypeError(
                        "Parameter dest ops should be a list of Operation, "
                        f"however {type(op)} was given."
                    )
                op.inputs.append(created)

        if source_op is not None:
            if not isinstance(source_op, Operation):
                raise TypeError(
                    "Parameter dest ops should be an Operation, "
                    f"however {type(source_op)} was given."
                )
            source_op.outputs.append(created)

        self.append_variable(created)
        return created

    def mark_variable_as_graph_input(self, var: Variable):
        r"""标记该变量为当前图的输入。

        Note:

            不能将输出标记为输入。
        """
        assert isinstance(var, Variable)
        var_name = var.name
        if var_name not in self.variables:
            raise KeyError(f"Can not find variable {var_name} within current graph.")
        if var_name in self.inputs:
            return
        if var_name in self.outputs:
            raise KeyError(
                f"Can not mark variable {var_name} as graph input, "
                "cause it is graph output."
            )
        self.inputs[var_name] = self.variables[var_name]

    def mark_variable_as_graph_output(self, var: Variable):
        r"""标记该变量为当前图的输出。

        Note:

            可以将输入标记为输出。
        """
        assert isinstance(var, Variable)
        var_name = var.name
        if var_name not in self.variables:
            raise KeyError(f"Can not find variable {var_name} within current graph.")
        if var_name in self.outputs:
            return
        self.outputs[var_name] = self.variables[var_name]

    def copy(self, copy_value: bool = False):  # noqa: C901
        """Clone current graph. Use parameter copy_value to control whether to
        do a Shallow Copy or Deep Copy.

        For copy_value = True, there will be a copy of each parameter in your network.
            ATTENTION: it might cause gpu memory overflow.
        For copy_value = False, cloned network will share the same parameter tensor of
            current one.

        ATTENTION: all quantization config will be cloned,
            all scales and offsets will be cloned even with copy_valye = False.

        Shallow Copy: Shallow repetition is quicker.
        However, it's “lazy” it handles pointers and references.
        Rather than creating a contemporary copy of the particular knowledge the
        pointer points to, it simply copies over the pointer price.
        So, each the first and therefore the copy can have pointers that reference
        constant underlying knowledge.

        Deep Copy: Deep repetition truly clones the underlying data.
        It is not shared between the first and therefore the copy.
        """

        cloned = BaseGraph(name=self.name)
        for op in self.operations.values():
            cloned.append_operation(deepcopy(op))
        for var in self.variables.values():
            cloned.append_variable(var.copy(copy_value=copy_value))

        # notice that all operations is copied without link, so do all variables
        # relink them with following code
        config_dict = {}
        for op in self.operations.values():
            assert (
                op.name in cloned.operations
            ), f"Graph Copy Error, Operation {op.name} is not correctly cloned"
            c_op = cloned.operations[op.name]
            for i_var in op.inputs:
                assert (
                    i_var.name in cloned.variables
                ), f"Graph Copy Error, Variable {i_var.name} is not correctly cloned"
                ci_var = cloned.variables[i_var.name]
                cloned.create_link_with_op(
                    variable=ci_var, op0=ci_var.source_op, op1=c_op
                )
            for o_var in op.outputs:
                assert (
                    o_var.name in cloned.variables
                ), f"Graph Copy Error, Variable {o_var.name} is not correctly cloned"
                co_var = cloned.variables[o_var.name]
                c_op.outputs.append(co_var)
                co_var.source_op = c_op
            if isinstance(op, QuantableOperation):
                for cfg, var in op.config_with_variable:
                    config_dict[cfg._hash] = (op, var)

        # relink config to there cloned master.
        for c_op in cloned.operations.values():
            if isinstance(c_op, QuantableOperation):
                for cfg, var in c_op.config_with_variable:
                    if cfg.dominated_by != cfg:
                        assert (
                            cfg.dominated_by._hash in config_dict
                        ), "Graph Copy Error, can not find a corresponding master config."
                        op, var = config_dict[cfg.dominated_by._hash]

                        op = cloned.operations[op.name]
                        assert isinstance(
                            op, QuantableOperation
                        ), "Graph Copy Error, Unexpected Master Operation Type."
                        for mcfg, mvar in op.config_with_variable:
                            if mvar.name == var.name:
                                cfg._dominator = mcfg

        # recreate input, output
        for name in self.inputs:
            cloned.inputs[name] = cloned.variables[name]
        for name in self.outputs:
            cloned.outputs[name] = cloned.variables[name]

        # check integrity
        for op in self.operations.values():
            if op.name not in cloned.operations:
                raise KeyError(f"Graph Copy Error, Operation {op.name} is Missing")
        for var in self.variables.values():
            if var.name not in cloned.variables:
                raise KeyError(f"Graph Copy Error, Variable {var.name} is Missing")
        for name in self.inputs:
            if name not in cloned.inputs:
                raise KeyError(f"Graph Copy Error, Input {name} is Missing")
        for name in self.outputs:
            if name not in cloned.outputs:
                raise KeyError(f"Graph Copy Error, Output {name} is Missing")
        cloned._num_of_generated_op = self._num_of_generated_op
        cloned._num_of_generated_var = self._num_of_generated_var
        cloned._detail = self._detail.copy()
        return cloned


class GraphBuilder(metaclass=SingletonMeta):
    r"""A base singleton for parser."""

    @abstractmethod
    def build(self, model_object: Any, **kwargs) -> BaseGraph:
        """Build BaseGraph from model_object, this function should be implemented"""


class GraphExporter(metaclass=SingletonMeta):
    r"""A base singleton for exporter."""

    @staticmethod
    def export_quantization_config(graph: BaseGraph):
        """Export Tensor Quantization Config (TQC) to file (JSON)."""

        render_buffer = {"configs": {}, "dispatchings": {}, "values": {}}
        # Render quantization config.
        for operation in graph.operations.values():
            op_dict = {}
            if isinstance(operation, QuantableOperation):
                for config, var in operation.config_with_variable:
                    op_dict[var.name] = dict(
                        bit_width=config.num_of_bits,
                        policy=config.policy.to_dict(),
                        state=config.state.name,
                        quant_min=config.quant_min,
                        quant_max=config.quant_max,
                        hash=hash(config),
                        dominator=hash(config.dominated_by),
                    )
                    if config.dominated_by == config:
                        if config.state != QuantizationStates.FP32:
                            render_buffer["values"][hash(config)] = {
                                "scale": _convert(config.scale),
                                "zero_point": _convert(config.offset),
                            }
            render_buffer["configs"][operation.name] = op_dict
            render_buffer["dispatchings"][operation.name] = operation.precision.name
        return render_buffer

    def dump_quantization_config(
        self, config_path: str | os.PathLike, graph: BaseGraph
    ):
        """Export Tensor Quantization Config (TQC) to file (JSON)."""
        render_buffer = self.export_quantization_config(graph)
        with open(file=config_path, mode="w", encoding="utf-8") as file:
            json.dump(render_buffer, file, indent=4)

    @abstractmethod
    def export(
        self,
        file_path: str | os.PathLike,
        graph: BaseGraph,
        config_path: Optional[str] = None,
        **kwargs,
    ):
        """Export BaseGraph to file_path, and export config to config_path if provided.
        This function should be implemented.
        """


class OperationExporter(metaclass=SingletonMeta):
    r"""A base singleton to transform an operation when exporting."""

    @abstractmethod
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        """Transform an operation to a new operation."""
