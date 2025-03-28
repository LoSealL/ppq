from hashlib import sha256
from itertools import product
from typing import Any, Dict, Iterator, List, Literal, Optional, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from mppq.common import (
    COMPUTING_OP,
    DEFAULT_OPSET_DOMAIN,
    DEFAULT_OPSET_VERSION,
    LINEAR_ACTIVATIONS,
    ONNX_DOMAIN,
)
from mppq.data import DataType, convert_any_to_tensor
from mppq.logger import warning
from mppq.quant import TargetPrecision
from mppq.register import Registry
from mppq.storage import Serializable


class Opset:
    """Open Neural Network Exchange (ONNX) is an open ecosystem that empowers
    AI developers to choose the right tools as their project evolves.

    ONNX provides an open source format for AI models, both deep learning and
    traditional ML.
    It defines an extensible computation graph model, as well as definitions of
    built-in operators and standard data types.
    Currently we focus on the capabilities needed for inferencing (scoring).

    PPQ IR is built based on ONNX definition.
    """

    def __init__(
        self,
        domain: str = DEFAULT_OPSET_DOMAIN,
        version: int = DEFAULT_OPSET_VERSION,
    ) -> None:
        self.domain = domain
        self.version = version

    def is_onnx(self):
        return self.domain == ONNX_DOMAIN


class Variable(Serializable):
    r"""定义网络中的变量。通常指的是ONNX中的常量算子（Constant）和初值（Initializer）"""

    def __init__(
        self,
        name: str,
        value: Optional[Tensor] = None,
        is_parameter: bool = False,
        dest_ops: Optional[List["Operation"]] = None,
        source_op: Optional["Operation"] = None,
        shape: Optional[Sequence[int | str]] = None,
        dtype: DataType = DataType.FP32,
    ) -> None:
        super().__init__()
        self.name = name
        self.is_parameter = is_parameter
        self.dest_ops = dest_ops or []
        self._source_op = source_op
        self._value = value
        self._shape = shape
        self._dtype = dtype

    @property
    def source_op(self) -> Optional["Operation"]:
        return self._source_op

    @source_op.setter
    def source_op(self, value: Optional["Operation"]):
        self._source_op = value

    @property
    def dest_idx(self) -> List[int]:
        _dest_idx = []
        for op in self.dest_ops:
            _dest_idx.append(op.inputs.index(self))
        return _dest_idx

    @property
    def src_idx(self) -> Optional[int]:
        if self._source_op is not None:
            return self.source_op.outputs.index(self)
        return None

    @property
    def value(self) -> Tensor:
        if self._value is None:
            raise ValueError(f"Variable {self.name} has no value.")
        return self._value

    @value.setter
    def value(self, value: Tensor):
        self._value = value
        self.shape = value.shape
        self.dtype = value.dtype

    @property
    def has_value(self) -> bool:
        return self._value is not None

    @property
    def shape(self) -> Sequence[str | int]:
        if self._shape is None:
            raise ValueError(f"Variable {self.name} has no shape.")
        return self._shape

    @shape.setter
    def shape(self, value: Sequence[int | str]):
        shape = [i for i in value]
        for element in shape:
            if not isinstance(element, (int, str)):
                raise TypeError(
                    f"Shape of a variable should only contains int or str. "
                    f"however {type(element)} was given."
                )
        self._shape = shape

    @property
    def has_shape(self) -> bool:
        return self._shape is not None

    @property
    def dtype(self) -> DataType:
        if self._dtype is None:
            raise ValueError(f"Variable {self.name} has no dtype.")
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if isinstance(value, np.dtype):
            self._dtype = DataType.from_numpy(value)
        elif isinstance(value, torch.dtype):
            self._dtype = DataType.from_torch(value)
        elif isinstance(value, DataType):
            self._dtype = value
        elif isinstance(value, str):
            self._dtype = DataType.from_numpy(np.dtype(value))
        else:
            raise TypeError(f"Invalid Dtype: {value} was given.")

    def copy(self, copy_value: bool = False):
        if not copy_value or not self.has_value:
            cloned = Variable(
                name=self.name,
                value=self.value,
                is_parameter=self.is_parameter,
                shape=self.shape,
                dtype=self.dtype,
            )
            return cloned

        if not isinstance(self.value, torch.Tensor):
            warning(
                f"You are requiring to copy variable {self.name}, "
                "however its value is not an instance of torch.Tensor, "
                "ppq will automatically convert it to torch.Tensor now."
            )
            self.value = convert_any_to_tensor(self.value)
        if isinstance(self.value, torch.Tensor):
            value = self.value.clone()
        else:
            value = self.value
        return Variable(
            name=self.name,
            value=value,
            is_parameter=self.is_parameter,
            shape=self.shape,
            dtype=self.dtype,
        )

    def __hash__(self) -> int:
        return int(sha256(self.name.encode("utf-8")).hexdigest(), 16)

    def __repr__(self) -> str:
        return f"{self.name}(shape={self.shape})"


class Operation(Serializable):
    def __init__(
        self,
        name: str,
        op_type: str,
        attributes: Dict[str, Any],
        precision: TargetPrecision = TargetPrecision.UNSPECIFIED,
        inputs: Optional[List[Variable]] = None,
        outputs: Optional[List[Variable]] = None,
        opset: Optional[Opset] = None,
    ) -> None:
        Serializable.__init__(self)
        self.name = name
        self.type = op_type
        self.attributes = attributes
        self.precision = precision
        if opset is None:
            self.opset = Opset()
        else:
            self.opset = opset
        self._detail = {}
        self._input_vars = inputs or []
        self._output_vars = outputs or []

    @property
    def inputs(self) -> List[Variable]:
        return self._input_vars

    @property
    def num_of_input(self) -> int:
        return len(self.inputs)

    @property
    def outputs(self) -> List[Variable]:
        return self._output_vars

    @property
    def num_of_output(self) -> int:
        return len(self.outputs)

    @property
    def is_computing_op(self) -> bool:
        return self.type in COMPUTING_OP

    @property
    def socket(self) -> "OpSocket":
        if self.type in DEFAULT_SOCKET_TABLE:
            return DEFAULT_SOCKET_TABLE[self.type](self)
        return DEFAULT_SOCKET_CREATOR(self)

    @property
    def parameters(self) -> List[Variable]:
        return [var for var in self.inputs if var.is_parameter]

    @property
    def num_of_parameter(self) -> int:
        return len(self.parameters)

    @property
    def is_linear_activation(self) -> bool:
        return self.type in LINEAR_ACTIVATIONS

    @property
    def is_boundary(self) -> bool:
        up_ops, down_ops = [], []
        for var in self.inputs:
            up_ops.append(var.source_op)
        for var in self.outputs:
            down_ops.extend(var.dest_ops)
        return all([op is None for op in up_ops]) or len(down_ops) == 0

    def set_extension_attrib(self, attrib: str, value: Any):
        self._detail[attrib] = value

    @property
    def extension_attrib(self) -> dict:
        return self._detail

    def __hash__(self) -> int:
        return int(sha256(self.name.encode("utf-8")).hexdigest(), 16)

    def __repr__(self) -> str:
        return (
            f"{self.name}(Type: {self.type}, "
            f"Num of Input: {self.num_of_input}, "
            f"Num of Output: {self.num_of_output})"
        )

    def __deepcopy__(self, memo=None) -> "Operation":
        clone = Operation(
            name=self.name,
            op_type=self.type,
            attributes=self.attributes.copy(),
            precision=self.precision,
            opset=self.opset,
        )
        clone._detail = self._detail.copy()
        return clone


class VLink:
    r"""定义一个算子的输入输出端口之间的计算关系。"""

    def __init__(self, in_idx: int, out_idx: int) -> None:
        self.in_idx = in_idx
        self.out_idx = out_idx

    def __getitem__(self, idx: Literal[0, 1]) -> int:
        return self.in_idx if idx == 0 else self.out_idx

    def __iter__(self) -> Iterator[int]:
        yield self.in_idx
        yield self.out_idx

    def __repr__(self) -> str:
        return f"VLink({self.in_idx}, {self.out_idx})"


class OpSocket:
    r"""定义一个算子的抽象计算关系。

    in_plat属性定义了该算子每个输入端口是否可被量化。在默认情况下，所有输入均会指定为 TargetPlatform.UNSPECIFIED
    可被量化。如果一些输入不可被量化，比如Resize的scale，roi，则需要指定为 TargetPlatform.FP32 或者其他平台。
    同理，out_plat属性定义了该算子每个输出端口是否可被量化。

    links属性定义了任意一对输入输出端口之间是否有计算关系。默认情况下，任意一对输入输出端口之间均存在计算关系。
    比如 Conv(x, weight, bias) 的输出需要x，weight，bias均参与计算，它们对输出的量化参数有依赖关系。
    而 Resize(x, roi, scales) 的输出不需要参roi, scales与计算，影响输出量化参数的只有x。
    """

    def __init__(
        self,
        op: Operation,
        in_plat: Optional[List[TargetPrecision]] = None,
        out_plat: Optional[List[TargetPrecision]] = None,
        links: Optional[List[VLink]] = None,
    ) -> None:
        self.in_plat = in_plat or [TargetPrecision.UNSPECIFIED] * op.num_of_input
        self.out_plat = out_plat or [TargetPrecision.UNSPECIFIED] * op.num_of_output
        self.links = links or [
            VLink(i, j)
            for i, j in product(range(op.num_of_input), range(op.num_of_output))
        ]


def _default_socket_creator(op: Operation) -> OpSocket:
    return OpSocket(op=op)


DEFAULT_SOCKET_CREATOR = _default_socket_creator


class _OpSocketFactory(Protocol):
    def __call__(self, op: Operation) -> OpSocket:
        return DEFAULT_SOCKET_CREATOR(op)


DEFAULT_SOCKET_TABLE: Registry[_OpSocketFactory] = Registry("DEFAULT_SOCKET_TABLE")
