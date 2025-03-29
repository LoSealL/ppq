from typing import Any, List, Optional

import torch

from mppq.data import convert_any_to_tensor
from mppq.ir.base.command import (
    GraphCommand,
    GraphCommandType,
    GraphDeployCommand,
    QuantizeOperationCommand,
    ReplaceOperationCommand,
    ReplaceVariableCommand,
)
from mppq.ir.base.graph import BaseGraph
from mppq.ir.base.opdef import Operation, Variable
from mppq.ir.base.processor import GraphCommandProcessor
from mppq.ir.base.quantize import QuantableOperation, QuantableVariable
from mppq.quant import OperationQuantizationConfig, TargetPrecision


class RunnableGraph(GraphCommandProcessor):
    r"""RunnableGraph deals with values related with graph executing.

    Literally it helps you move values of your graph towards device and vice versa.
        And give an executable order of all operations in your graph which actual
        executor will follow.

    Args:
        graph (BaseGraph): BaseGraph instance.
        device (str, optional): This attribute is only used by with
            RunnableGraph(graph, device) syntactic.
    """

    def __init__(self, graph: BaseGraph, device: Optional[str] = None) -> None:
        super().__init__(graph_or_processor=graph)
        self._device = device or "cpu"

    def process(self, command: GraphCommand):
        if command.command_type == GraphCommandType.DEPLOY_TO_CPU:
            return self.deploy("cpu")
        elif command.command_type == GraphCommandType.DEPLOY_TO_CUDA:
            if isinstance(command, GraphDeployCommand):
                device = command.device
                return self.deploy(device)
            else:
                return self.deploy("cuda")
        raise RuntimeError

    @property
    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [GraphCommandType.DEPLOY_TO_CPU, GraphCommandType.DEPLOY_TO_CUDA]

    def deploy(self, device: str):
        for _, operator in self._graph.operations.items():
            assert isinstance(operator, Operation)
            # in onnx format, some constant values are wrapped with operation's
            # attributes['value']. To move those constant value from numpy to device,
            # we have to move all the attributes['value'] of operation to device.
            if (
                operator.type == "Constant"
                and operator.precision != TargetPrecision.SOI
            ):
                operator.attributes["value"] = convert_any_to_tensor(
                    operator.attributes["value"], device=device
                )
            if (
                operator.type == "Constant"
                and operator.precision == TargetPrecision.SOI
            ):
                operator.attributes["value"] = convert_any_to_tensor(
                    operator.attributes["value"], device="cpu"
                )
            # PATCH 20220706, send quantization config to device.
            if isinstance(operator, QuantableOperation):
                for cfg, _ in operator.config_with_variable:
                    # pylint: disable=protected-access
                    if isinstance(cfg._scale, torch.Tensor):
                        cfg._scale = cfg._scale.to(device)
                    if isinstance(cfg._offset, torch.Tensor):
                        cfg._offset = cfg._offset.to(device)

        for _, variable in self._graph.variables.items():
            assert isinstance(variable, Variable)
            # graph output variable has no destinations
            if len(variable.dest_ops) == 0:
                continue
            if not variable.has_value:
                continue

            # check all destination operations platform are same.
            precisions = [op.precision for op in variable.dest_ops]
            if (
                all([_ == precisions[0] for _ in precisions])
                and precisions[0] == TargetPrecision.SOI
            ):
                precision = TargetPrecision.SOI
            else:
                precision = TargetPrecision.UNSPECIFIED

            # if all downstream operations are shape related operations,
            # send value to cpu
            if precision == TargetPrecision.SOI:
                variable.value = convert_any_to_tensor(variable.value).to("cpu")
            else:
                variable.value = convert_any_to_tensor(variable.value).to(device=device)

            # if variable is a shape-related variable, send it to cpu.
            if variable.is_parameter:
                if len(variable.dest_ops) > 1:
                    raise PermissionError(
                        f"PPQ can not process parameter variable({variable.name})"
                        " with multiple destinations"
                        f"({[op.name for op in variable.dest_ops]}), split it first."
                    )
                dest_op = variable.dest_ops[0]
                dest_idx = dest_op.inputs.index(variable)

                assert isinstance(dest_op, Operation)
                socket = dest_op.socket
                if socket.in_plat[dest_idx] == TargetPrecision.SOI:
                    variable.value = convert_any_to_tensor(variable.value, device="cpu")
        return self


class QuantableGraph(GraphCommandProcessor):
    def process(self, command: GraphCommand) -> Any:
        if command.command_type == GraphCommandType.QUANTIZE_OPERATION:
            assert isinstance(command, QuantizeOperationCommand)
            return self.quantize_operation(
                command.op_name, command.target_precision, command.config
            )

    @property
    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [GraphCommandType.QUANTIZE_OPERATION]

    def quantize_operation(
        self,
        operation_name: str,
        target_precision: TargetPrecision,
        quantization_config: OperationQuantizationConfig,
    ) -> QuantableOperation:
        if operation_name not in self.graph.operations:
            raise KeyError(
                f"Operation {operation_name} is not in your graph, "
                "Please check your input."
            )

        operation = self._graph.operations[operation_name]
        quantized_operation = QuantableOperation(
            convert_from=operation,
            quantize_config=quantization_config,
            platform=target_precision,
        )

        # calling other chain responder to replace operation with quantized one.
        if self._next_command_processor is None:
            raise RuntimeError(
                "To replace a operation, your processor chain must have "
                "a GraphReplacer Processor."
            )
        self._next_command_processor(
            ReplaceOperationCommand(operation_name, quantized_operation)
        )

        # replace all related variable with quantable one.
        for var in quantized_operation.inputs + quantized_operation.outputs:
            if isinstance(var, QuantableVariable):
                continue
            self._next_command_processor(
                ReplaceVariableCommand(
                    var_name=var.name, replace_to=QuantableVariable(convert_from=var)
                )
            )
        quantized_operation.store_parameter_value()
        return quantized_operation

    def dequantize_operation(self, operation_name: str) -> Operation:
        if operation_name not in self.graph.operations:
            raise KeyError(
                f"Operation {operation_name} is not in your graph, "
                "Please check your input."
            )
        operation = self._graph.operations[operation_name]
        if not isinstance(operation, QuantableOperation):
            return operation
        else:
            return operation.dequantize()

    def dequantize_graph(self, expire_device: str = "cpu"):
        """一个方便懒人的函数."""
        for operation in self.graph.operations.values():
            if isinstance(operation, QuantableOperation):
                operation.dequantize(expire_device=expire_device)

    def restore_quantize_state(self, expire_device: str = "cpu"):
        """一个方便懒人的函数."""
        for operation in self.graph.operations.values():
            if isinstance(operation, QuantableOperation):
                operation.restore_quantize_state(expire_device=expire_device)
