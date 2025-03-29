"""
Copyright Wenyi Tang 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import tempfile
from enum import IntEnum

import onnx
import pytest
import torch

from mppq.api import ENABLE_CUDA_KERNEL, quantize, register_platform


class TestPlatform(IntEnum):
    MY_PLATFORM = 1
    TEST1 = 999
    TEST2 = 1000
    TEST3 = 1001


register_platform(TestPlatform.MY_PLATFORM, {"allin": None}, {})


def test_register_platform():
    register_platform(TestPlatform.TEST1, {"allin": None}, {})
    with pytest.raises(KeyError):
        register_platform(TestPlatform.TEST1, {}, {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_simple_quantize(model):
    with ENABLE_CUDA_KERNEL(), tempfile.TemporaryDirectory() as tmpdir:
        quantize(model, f"{tmpdir}/test.onnx", TestPlatform.MY_PLATFORM)
        model = onnx.load(f"{tmpdir}/test.onnx")
        onnx.checker.check_model(model, True)
