"""PPQ Core File System PPQ 核心文件系统(IO)

You are not allowed to modify this 请勿修改此文件
"""

import pickle
from typing import Any

import numpy as np
import torch

import mppq
from mppq.data import DataType, convert_any_to_numpy, convert_any_to_tensor
from mppq.logger import warning


class Serializable:
    """An interface which means a class instance is binary serializable,
    nothing funny."""

    def __init__(self) -> None:
        self._export_value = False

    def __setstate__(self, state: dict):
        if not isinstance(state, dict):
            raise TypeError(
                f"PPQ Data Load Failure. Can not load data from {type(state)}, "
                "Your data might get damaged."
            )

        if "__version__" not in state or state["__version__"] != mppq.__version__:
            warning(
                "You are loading an object created by PPQ with different version,"
                " it might cause some problems."
            )

        for key, value in state.items():
            self.__dict__[key] = value
            if isinstance(value, ValueState):
                self.__dict__[key] = value.unpack()
        return self

    def __getstate__(self) -> dict:
        attribute_dicts = self.__dict__
        attribute_dicts["__version__"] = mppq.__version__
        serialized = dict()

        for name, value in attribute_dicts.items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                if self._export_value is False:
                    value = None
                serialized[name] = ValueState(value)
            else:
                serialized[name] = value
        return serialized


class ValueState(Serializable):
    def __init__(self, value: Any) -> None:
        self._value: Any
        self._shape: Any
        self._value_type = str(value.__class__.__name__)
        if isinstance(value, np.ndarray):
            self._dtype = value.dtype
            self._shape = value.shape
            self._value = pickle.dumps(value)
        elif isinstance(value, torch.Tensor):
            self._dtype = DataType.to_numpy(DataType.from_torch(value.dtype))
            self._shape = value.shape
            self._value = pickle.dumps(convert_any_to_numpy(value))
        elif isinstance(value, list) or isinstance(value, tuple):
            self._value = value
            self._dtype = None
            self._shape = None
        elif value is None:
            self._dtype = None
            self._shape = None
            self._value = None
        else:
            raise TypeError(
                f"PPQ Data Dump Failure, can not dump value type {type(value)}"
            )

    def unpack(self) -> Any:
        if self._value_type == str(None.__class__.__name__):
            return None
        elif self._value_type == str("ndarray"):
            value = pickle.loads(self._value)
            assert isinstance(value, np.ndarray)
            value = value.astype(self._dtype)
            value = value.reshape(self._shape)
            return value
        elif self._value_type == str("Tensor"):
            if self._value is not None:
                value = pickle.loads(self._value)
                assert isinstance(value, np.ndarray)
                value = value.astype(self._dtype)
                if value is not None:
                    value = value.reshape(self._shape)
                value = convert_any_to_tensor(
                    value,
                    device="cpu",
                    dtype=DataType.to_torch(
                        DataType.from_numpy(self._dtype),  # type: ignore
                    ),
                )
                return value
            else:
                return torch.tensor([], device="cpu")
        elif self._value_type in {"list", "tuple", "dict"}:
            return self._value
