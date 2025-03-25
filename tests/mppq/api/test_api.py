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

from mppq.api import (
    ENABLE_CUDA_KERNEL,
    export_onnx_graph,
    load_quantizer,
    register_platform,
)


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
    q = load_quantizer(model, TestPlatform.MY_PLATFORM)
    with ENABLE_CUDA_KERNEL():
        g = q.quantize()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_onnx_graph(g, f"{tmpdir}/test.onnx")
        model = onnx.load(f"{tmpdir}/test.onnx")
        onnx.checker.check_model(model, True)
