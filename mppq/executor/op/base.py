# pylint: disable=invalid-name

from typing import Any, List, Optional, Protocol, Sequence, Tuple

import torch

from mppq.ir.base.opdef import Operation
from mppq.ir.base.quantize import QuantableOperation
from mppq.quant import TargetPrecision
from mppq.register import Registry


class TorchBackendContext:
    def __init__(self, executing_device: str) -> None:
        self.executing_device = executing_device


def ASSERT_NUM_OF_INPUT(
    op: Operation,
    values: Sequence[torch.Tensor],
    min_num_of_input: int = -1,
    max_num_of_input: int = 99,
):
    if min_num_of_input == max_num_of_input:
        if len(values) != min_num_of_input:
            raise ValueError(
                f"Can not feed value to operation {op.name}, "
                f"expects exact {min_num_of_input} inputs, "
                f"however {len(values)} was given"
            )
    elif len(values) > max_num_of_input:
        raise ValueError(
            f"Too many input value for {op.name}, "
            f"expects {max_num_of_input} inputs at most, "
            f"however {len(values)} was given"
        )
    elif len(values) < min_num_of_input:
        raise ValueError(
            f"Too few input value for {op.name}, "
            f"expects {min_num_of_input} inputs at least, "
            f"however {len(values)} was given"
        )


def GET_ATTRIBUTE_FROM_OPERATION(
    op: Operation, attribute: str, compulsive: bool = False, default: Any = None
):
    """Try to get an attribute from operation. If an attribute is compulsive,
    then operation must give a value of it, otherwise an error will be thrown.
    If an attribute is not compulsive, a default value will be given if
    operation.attributes do not holds a value of requesting attribute.

    Args:
        op (Operation): Operation instance.
        attribute (str): Attribute name.
        compulsive (bool): Whether is a compulsive attribute.
        default (Any, optional): [description]. default value of attribute.
    """
    if attribute in op.attributes:
        return op.attributes[attribute]
    else:
        if compulsive:
            raise KeyError(
                f"Operation {op.name} is supposed to have a value of attribute "
                f"{attribute}. However this value is missing from currecnt operation.",
            )
        else:
            return default


def GET_VALUE_FROM_INPUTS(
    values: Sequence[torch.Tensor], idx: int
) -> Optional[torch.Tensor]:
    assert isinstance(idx, int) and idx >= 0
    if len(values) > idx:
        return values[idx]
    else:
        return None


def ASSERT_IS_QUANT_OP(op):
    if not isinstance(op, QuantableOperation):
        raise TypeError(
            "Given Operation is expected as a QuantableOperation, "
            f"however {type(op)} was given."
        )


def FORCE_CONVERT_DEVICE(
    value: torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    return value.to(device=device, copy=True)


def VALUE_TO_EXECUTING_DEVICE(
    op: Operation, ctx: Optional[TorchBackendContext], values: Sequence[torch.Tensor]
) -> List[torch.Tensor]:
    values = list(values)
    if ctx is None:
        device = values[0].device
    else:
        device = ctx.executing_device
    for idx, (plat, value) in enumerate(zip(op.socket.in_plat, values)):
        if value is None:
            continue
        if plat == TargetPrecision.SOI or op.precision == TargetPrecision.SOI:
            values[idx] = value.cpu()
        else:
            values[idx] = value.to(device)
    return values


class OperationForwardProtocol(Protocol):
    """A protocol for operation forward function."""

    def __call__(
        self,
        op: Operation,
        values: Sequence[torch.Tensor],
        ctx: Optional[TorchBackendContext] = None,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Args:
            op: operation information (precision, attributes, types, etc.)
            values: input tensors
            ctx: execution device information

        Returns:
            a result tensor or a tuple of result tensors
        """
        raise NotImplementedError


DEFAULT_BACKEND_TABLE: Registry[OperationForwardProtocol] = Registry("BACKEND_TABLE")
