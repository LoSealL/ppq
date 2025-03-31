"""
Copyright Wenyi Tang 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest
import torch

from mppq.ir.base.opdef import Variable
from mppq.quant import (
    QuantizationPolicy,
    QuantizationProperty,
    TensorQuantizationConfig,
)
from mppq.quantization.observer.base import OBSERVER_TABLE


def per_tensor_affine():
    policy = QuantizationPolicy(
        QuantizationProperty.ASYMMETRIC
        | QuantizationProperty.LINEAR
        | QuantizationProperty.PER_TENSOR,
    )
    cfg = TensorQuantizationConfig(policy, quant_min=0, quant_max=255)
    return cfg


def per_tensor_symmetric():
    policy = QuantizationPolicy(
        QuantizationProperty.SYMMETRIC
        | QuantizationProperty.LINEAR
        | QuantizationProperty.PER_TENSOR,
    )
    cfg = TensorQuantizationConfig(policy, quant_min=-128, quant_max=127)
    return cfg


SIZE = 1000


def gaussian_dist(seed=42, mean=0, sigma=0.2):
    """Fixed gaussian distribution"""
    np.random.seed(seed)

    for _ in range(SIZE):
        yield np.random.normal(mean, sigma, size=[32])
    np.random.seed(None)


def gamma_dist(seed=42, shape=1, sigma=0.2):
    np.random.seed(seed)

    for _ in range(SIZE):
        yield np.random.gamma(shape, sigma, size=[32])
    np.random.seed(None)


def beta_dist(seed=42, alpha=0.2, beta=0.2):
    np.random.seed(seed)

    for _ in range(SIZE):
        yield np.random.beta(alpha, beta, size=[32])
    np.random.seed(None)


def chisquare_dist(seed=42, df=0.1):
    np.random.seed(seed)

    for _ in range(SIZE):
        yield np.random.chisquare(df, size=[32])
    np.random.seed(None)


@pytest.mark.parametrize("dist", [gaussian_dist, gamma_dist, beta_dist, chisquare_dist])
def test_all_observers_per_tensor_symmetric(dist):
    loss = {}
    for key in OBSERVER_TABLE:
        cls = OBSERVER_TABLE[key]
        try:
            observer = cls(Variable("a"), per_tensor_symmetric())
            for value in dist():
                observer.observe(torch.tensor(value).float())
            observer.render_quantization_config()
            # second phase, for observer without 2nd phase, do nothing
            for value in dist():
                observer.observe(torch.tensor(value).float())
            observer.render_quantization_config()
        except (TypeError, PermissionError):
            continue
        mse = 0
        for value in dist():
            scale = float(observer._quant_cfg.scale)
            zero_point = int(observer._quant_cfg.offset)
            qvalue = np.clip(
                np.round(value / scale) + zero_point,
                observer._quant_cfg.quant_min,
                observer._quant_cfg.quant_max,
            )
            qvalue = (qvalue - zero_point) * scale
            mse += np.sum((qvalue - value) ** 2)
        mse /= 10000
        loss[key] = mse
    print(f"\ntest_all_observers_per_tensor_symmetric {dist.__name__}")
    for name, mse in sorted(loss.items(), key=lambda pair: pair[1]):
        print(f"{name}: {mse}")
    print("")


@pytest.mark.parametrize("dist", [gaussian_dist, gamma_dist, beta_dist, chisquare_dist])
def test_possible_observers_per_tensor_affine(dist):
    loss = {}
    for key in OBSERVER_TABLE:
        cls = OBSERVER_TABLE[key]
        try:
            observer = cls(Variable("a"), per_tensor_affine())
            for value in dist():
                observer.observe(torch.tensor(value).float())
            observer.render_quantization_config()
            # second phase, for observer without 2nd phase, do nothing
            for value in dist():
                observer.observe(torch.tensor(value).float())
            observer.render_quantization_config()
        except (TypeError, PermissionError):
            continue
        mse = 0
        for value in dist():
            scale = float(observer._quant_cfg.scale)
            zero_point = int(observer._quant_cfg.offset)
            qvalue = np.clip(
                np.round(value / scale) + zero_point,
                observer._quant_cfg.quant_min,
                observer._quant_cfg.quant_max,
            )
            qvalue = (qvalue - zero_point) * scale
            mse += np.sum((qvalue - value) ** 2)
        mse /= 10000
        loss[key] = mse
    print(f"\ntest_possible_observers_per_tensor_affine {dist.__name__}")
    for name, mse in sorted(loss.items(), key=lambda pair: pair[1]):
        print(f"{name}: {mse}")
    print("")
