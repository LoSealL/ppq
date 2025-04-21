"""
Copyright Wenyi Tang 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import pickle
from copy import deepcopy

import pytest
import torch

from mppq.quant import (
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
)


class TestQuantizationPolicy:
    def test_policy_combine(self):
        with pytest.raises(AssertionError):
            QuantizationPolicy(QuantizationProperty.DYNAMIC)
        with pytest.raises(AssertionError):
            QuantizationPolicy(
                QuantizationProperty.LINEAR | QuantizationProperty.FLOATING
            )
        QuantizationPolicy(
            QuantizationProperty.LINEAR
            | QuantizationProperty.PER_CHANNEL
            | QuantizationProperty.SYMMETRIC
        )

    def test_policy_to_dict(self):
        policy = QuantizationPolicy(
            QuantizationProperty.LINEAR
            | QuantizationProperty.PER_TENSOR
            | QuantizationProperty.ASYMMETRIC
        ).to_dict()

        assert len(policy) == len(QuantizationProperty)
        assert policy[QuantizationProperty.LINEAR.name]
        assert policy[QuantizationProperty.PER_TENSOR.name]
        assert policy[QuantizationProperty.ASYMMETRIC.name]
        assert not policy[QuantizationProperty.PER_CHANNEL.name]
        assert not policy[QuantizationProperty.FLOATING.name]
        assert not policy[QuantizationProperty.SYMMETRIC.name]
        assert not policy[QuantizationProperty.DYNAMIC.name]

    def test_policy_equality_check(self):
        p1 = QuantizationPolicy(
            QuantizationProperty.LINEAR
            | QuantizationProperty.PER_CHANNEL
            | QuantizationProperty.SYMMETRIC
        )
        p2 = QuantizationPolicy(
            QuantizationProperty.LINEAR
            | QuantizationProperty.PER_CHANNEL
            | QuantizationProperty.SYMMETRIC
        )
        p3 = QuantizationPolicy(
            QuantizationProperty.LINEAR
            | QuantizationProperty.PER_CHANNEL
            | QuantizationProperty.ASYMMETRIC
        )
        assert p1 == p2
        assert p1 != p3

        with pytest.raises(TypeError):
            assert p1 == 1


class TestQuantizationStates:
    def test_is_activated(self):
        assert not QuantizationStates.is_activated(QuantizationStates.INITIAL)
        assert not QuantizationStates.is_activated(QuantizationStates.BAKED)
        assert not QuantizationStates.is_activated(QuantizationStates.OVERLAPPED)
        assert not QuantizationStates.is_activated(QuantizationStates.PASSIVE_INIT)
        assert not QuantizationStates.is_activated(QuantizationStates.PASSIVE_BAKED)
        assert not QuantizationStates.is_activated(QuantizationStates.FP32)
        assert QuantizationStates.is_activated(QuantizationStates.ACTIVATED)
        assert QuantizationStates.is_activated(QuantizationStates.PASSIVE)

    def test_can_export(self):
        assert not QuantizationStates.can_export(QuantizationStates.INITIAL)
        assert not QuantizationStates.can_export(QuantizationStates.PASSIVE_INIT)
        assert QuantizationStates.can_export(QuantizationStates.BAKED)
        assert QuantizationStates.can_export(QuantizationStates.OVERLAPPED)
        assert QuantizationStates.can_export(QuantizationStates.ACTIVATED)
        assert QuantizationStates.can_export(QuantizationStates.PASSIVE)
        assert QuantizationStates.can_export(QuantizationStates.FP32)


class TestTensorQuantizationConfig:
    def test_create_hash(self):
        config = TensorQuantizationConfig(QuantizationPolicy.ALC())
        assert hash(config) == config._hash

    def test_serialize(self):
        config = TensorQuantizationConfig(QuantizationPolicy.ALC())
        bins = pickle.dumps(config)
        config_restore = pickle.loads(bins)
        assert config == config_restore

    def test_is_same_scheme(self):
        config1 = TensorQuantizationConfig(QuantizationPolicy.SLC())
        config2 = TensorQuantizationConfig(QuantizationPolicy.SLC())
        assert config1 != config2
        assert config1.is_same_scheme(config2)
        assert config2.is_same_scheme(config1)

    def test_scale(self):
        config_with_scale = TensorQuantizationConfig(
            QuantizationPolicy.ALC(), scale=torch.Tensor([1])
        )
        torch.testing.assert_close(config_with_scale.scale, torch.Tensor([1]))
        config_with_scale.scale = torch.Tensor([2])
        torch.testing.assert_close(config_with_scale.scale, torch.Tensor([2]))

        config_no_scale = TensorQuantizationConfig(QuantizationPolicy.ALC())
        with pytest.raises(ValueError):
            config_no_scale.scale
        config_with_scale.scale = None
        with pytest.raises(ValueError):
            config_with_scale.scale

    def test_dominated_by(self):
        master_config = TensorQuantizationConfig(QuantizationPolicy.ALC())
        assert master_config.dominated_by == master_config
        master_config.scale = torch.Tensor([1])
        master_config.offset = torch.Tensor([0])
        slave_config = TensorQuantizationConfig(QuantizationPolicy.ALC())
        slave_config.dominated_by = master_config

        assert slave_config.state == QuantizationStates.OVERLAPPED
        torch.testing.assert_close(slave_config.scale, torch.Tensor([1]))
        torch.testing.assert_close(slave_config.offset, torch.Tensor([0]))

        master_master = TensorQuantizationConfig(
            QuantizationPolicy.ALC(), scale=torch.Tensor([2]), offset=torch.Tensor([3])
        )
        master_config.dominated_by = master_master
        torch.testing.assert_close(slave_config.scale, torch.Tensor([2]))
        torch.testing.assert_close(slave_config.offset, torch.Tensor([3]))

        with pytest.raises(ValueError):
            # cyclic dependency
            master_master.dominated_by = slave_config
        with pytest.raises(ValueError):
            # recover to self
            master_config.dominated_by = master_config

    def test_master_by(self):
        master_config = TensorQuantizationConfig(QuantizationPolicy.ALC())
        assert master_config.master_by == master_config
        master_config.scale = torch.Tensor([1])
        master_config.offset = torch.Tensor([0])
        slave_config = TensorQuantizationConfig(QuantizationPolicy.ALC())
        slave_config.master_by = master_config

        assert slave_config.state == QuantizationStates.PASSIVE
        torch.testing.assert_close(slave_config.scale, torch.Tensor([1]))
        torch.testing.assert_close(slave_config.offset, torch.Tensor([0]))

        master_master = TensorQuantizationConfig(
            QuantizationPolicy.ALC(), scale=torch.Tensor([2]), offset=torch.Tensor([3])
        )
        master_config.master_by = master_master
        torch.testing.assert_close(slave_config.scale, torch.Tensor([2]))
        torch.testing.assert_close(slave_config.offset, torch.Tensor([3]))

        # cyclic dependency
        master_master.master_by = slave_config
        assert master_master.master_by == master_master
        # recover to self
        master_config.master_by = master_config
        torch.testing.assert_close(master_config.scale, torch.Tensor([1]))
        torch.testing.assert_close(master_config.offset, torch.Tensor([0]))
        assert master_config.state == QuantizationStates.ACTIVATED

    def test_deepcopy(self):
        master_config = TensorQuantizationConfig(QuantizationPolicy.ALC())
        master_config.scale = torch.Tensor([1])
        master_config.offset = torch.Tensor([0])
        slave_config = TensorQuantizationConfig(QuantizationPolicy.ALC())
        slave_config.dominated_by = master_config
        master_master = TensorQuantizationConfig(QuantizationPolicy.ALC())
        master_config.dominated_by = master_master

        copy_config = deepcopy(slave_config)
        assert copy_config != slave_config
        assert copy_config.is_same_scheme(slave_config)


class TestOperationQuantizationConfig:
    def test_iter(self):
        in_cfg = [
            TensorQuantizationConfig(QuantizationPolicy.SLT()),
            TensorQuantizationConfig(QuantizationPolicy.ALC()),
        ]
        out_cfg = [
            TensorQuantizationConfig(QuantizationPolicy.SLT()),
        ]
        config = OperationQuantizationConfig(in_cfg, out_cfg)
        assert len(config) == 3
        for i, j in zip(config, in_cfg + out_cfg):
            assert isinstance(i, TensorQuantizationConfig)
            assert i == j

    def test_deepcopy(self):
        in_cfg = [
            TensorQuantizationConfig(QuantizationPolicy.SLT()),
            TensorQuantizationConfig(QuantizationPolicy.ALC()),
        ]
        out_cfg = [
            TensorQuantizationConfig(QuantizationPolicy.SLT()),
        ]
        config = OperationQuantizationConfig(in_cfg, out_cfg)
        copy_config = deepcopy(config)
        for i, j in zip(copy_config, in_cfg + out_cfg):
            assert isinstance(i, TensorQuantizationConfig)
            assert i != j
            assert i.is_same_scheme(j)
