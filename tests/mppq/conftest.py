"""
Copyright Wenyi Tang 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Config pytest
"""

from pathlib import Path

import pytest

model_dir = Path(__file__).parent.parent / "models"


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Generate parametrized arguments to all tests with arg 'model'."""

    models = sorted(model_dir.rglob("*.onnx"))
    if "model" in metafunc.fixturenames:
        metafunc.parametrize("model", models)
