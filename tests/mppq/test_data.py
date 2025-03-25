import numpy as np
import pytest
import torch

from mppq.data import (
    DataType,
    convert_any_to_numpy,
    convert_any_to_python_primary_type,
    convert_any_to_tensor,
)


# pylint: disable=missing-function-docstring
@pytest.mark.parametrize(
    "dtype",
    [
        np.bool,
        np.uint8,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
        np.dtype(np.bool),
        np.dtype(np.uint8),
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    ],
)
def test_data_type_from_numpy(dtype):
    mytype = DataType.from_numpy(dtype)
    assert DataType.to_numpy(mytype) == dtype


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.float8_e5m2,
    ],
)
def test_data_type_from_torch(dtype):
    mytype = DataType.from_torch(dtype)
    assert DataType.to_torch(mytype) == dtype


@pytest.mark.parametrize(
    ("value", "accept_none", "expected"),
    [
        (1, False, np.array([1])),
        ([1, 2, 3], False, np.array([1, 2, 3])),
        ((1, 2, 3), False, np.array((1, 2, 3))),
        ("123", False, np.array("123")),
        (np.array([1, 2, 3]), False, np.array([1, 2, 3])),
        (np.empty(0), False, ValueError),
        (np.empty(0), True, None),
        (np.array(1), False, np.array(1)),
        (torch.tensor([]), False, ValueError),
        (torch.tensor([]), True, None),
        (torch.tensor(1), False, np.array([1])),
        (torch.tensor([1, 2, 3]), False, np.array([1, 2, 3])),
    ],
)
def test_convert_any_to_numpy(value, accept_none, expected):
    if isinstance(expected, type(object)) and issubclass(expected, Exception):
        with pytest.raises(expected):
            convert_any_to_numpy(value, accept_none=accept_none)
    else:
        np_value = convert_any_to_numpy(value, accept_none=accept_none)
        assert np.all(np_value == expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, torch.tensor([1])),
        ([1, 2, 3], torch.tensor([1, 2, 3])),
        ((1, 2, 3), torch.tensor((1, 2, 3))),
        ("123", TypeError),
        (np.array([1, 2, 3]), torch.tensor([1, 2, 3])),
        (np.empty(0), torch.tensor([])),
        (np.empty(0), torch.tensor([])),
        (np.array(1), torch.tensor(1)),
        (torch.tensor([]), torch.tensor([])),
        (torch.tensor([]), torch.tensor([])),
        (torch.tensor(1), torch.tensor([1])),
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])),
    ],
)
def test_convert_any_to_tensor(value, expected):
    if isinstance(expected, type(object)) and issubclass(expected, Exception):
        with pytest.raises(expected):
            convert_any_to_tensor(value)
    else:
        tensor_value = convert_any_to_tensor(value)
        assert torch.all(tensor_value == expected)


@pytest.mark.parametrize(
    ("value", "accept_none", "expected"),
    [
        (np.array([1, 2, 3]), False, [1, 2, 3]),
        (torch.tensor([1, 2, 3]), False, [1, 2, 3]),
        (np.array([1]), False, 1),
        (torch.tensor([1]), False, 1),
        (np.empty(0), False, ValueError),
        (np.empty(0), True, None),
        (torch.empty(0), False, ValueError),
        (torch.empty(0), True, None),
        (1, False, 1),
        (None, False, ValueError),
        (None, True, None),
        ([1, 2, 3], False, [1, 2, 3]),
    ],
)
def test_convert_any_to_python_primary_type(value, accept_none, expected):
    if isinstance(expected, type(object)) and issubclass(expected, Exception):
        with pytest.raises(expected):
            convert_any_to_python_primary_type(value, accept_none=accept_none)
    else:
        python_value = convert_any_to_python_primary_type(
            value, accept_none=accept_none
        )
        assert python_value == expected
