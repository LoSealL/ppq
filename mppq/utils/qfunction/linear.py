from typing import Tuple

import torch
from torch.autograd import Function

from mppq.common import OBSERVER_MIN_SCALE, OBSERVER_MIN_SCALE_MANUL_OVERRIDE
from mppq.ffi import CUDA, ENABLE_CUDA_KERNEL
from mppq.logger import warning
from mppq.quant import QuantizationProperty, RoundingPolicy, TensorQuantizationConfig
from mppq.utils.round import (
    ppq_numerical_round,
    ppq_round_to_power_of_2,
    ppq_tensor_round,
)


def minmax_to_scale_offset(
    min_val: float,
    max_val: float,
    config: TensorQuantizationConfig,
    scale_threshold: float = OBSERVER_MIN_SCALE,
) -> Tuple[float, float]:
    """
    Solve scale and offset with given min, max value.
    For Symmetrical Quantization, offset is set to 0.
    For ASymmetrical Quantization, offset is limited by
    [config.quant_min, config.quant_max].

    Scale is limited by [scale_threshold, +inf].

    Args:
        min_val (float): min value
        max_val (float): max value
        config (TensorQuantizationConfig): Corresponding TQC.
        scale_threshold (float, optional): minimum scale.

    Returns:
        Tuple[float, float]: Solved scale and offset.
    """
    if OBSERVER_MIN_SCALE_MANUL_OVERRIDE in config.detail:
        scale_threshold = config.detail[OBSERVER_MIN_SCALE_MANUL_OVERRIDE]

    scale, offset = 1, 0
    if min_val > 0:
        min_val = 0
    if max_val < 0:
        max_val = 0

    if config.policy.has_property(QuantizationProperty.ASYMMETRIC):
        val_range = float(max_val - min_val)
        scale = val_range / (config.quant_max - config.quant_min)
        if scale < scale_threshold:
            warning(
                "Numeric instability detected: "
                "ppq find there is a scale value < 1e-7, "
                "which probably cause numeric underflow in further computation."
            )
        scale = max(scale, scale_threshold)
        offset = ppq_numerical_round(-min_val / scale)
    elif config.policy.has_property(QuantizationProperty.SYMMETRIC):
        val_range = 2 * float(max(abs(max_val), abs(min_val)))
        scale = val_range / (config.quant_max - config.quant_min)
        if scale < scale_threshold:
            warning(
                "Numeric instability detected: "
                "ppq find there is a scale value < 1e-7, "
                "which probably cause numeric underflow in further computation."
            )
        scale = max(scale, scale_threshold)
        offset = 0
    else:
        raise TypeError(
            "Tensor Min Max Observer Excepts either ASYMMETRICAL or SYMMETRICAL "
            "quantization config."
        )
    if config.policy.has_property(QuantizationProperty.POWER_OF_2):
        scale = ppq_round_to_power_of_2(scale, policy=RoundingPolicy.ROUND_UP)
    return scale, offset


# pylint: disable=abstract-method, arguments-differ
class TensorwiseLinearQuantImpl(Function):
    r"""Torch Tensorwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will use ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        quant_min: int,
        quant_max: int,
        rounding: RoundingPolicy,
    ) -> torch.Tensor:
        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)

        if not ENABLE_CUDA_KERNEL.USING_CUDA_KERNEL or not tensor.is_cuda:
            # quantization function, pytorch implementation
            tensor = ppq_tensor_round((tensor / scales), rounding) + offsets
            tensor = torch.clamp(tensor, quant_min, quant_max)
            tensor = (tensor - offsets) * scales
            return tensor
        else:
            # quantization function, pure cuda implementation
            quantized = CUDA.LinearQuantize_T(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value,
            )
            return quantized

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        return grad_outputs[0], None, None, None, None, None, None, None, None


class ChannelwiseLinearQuantImpl(Function):
    """Torch Channelwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        quant_min: int,
        quant_max: int,
        rounding: RoundingPolicy,
    ) -> torch.Tensor:
        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)

        if not ENABLE_CUDA_KERNEL.USING_CUDA_KERNEL or not tensor.is_cuda:
            # generate a shape that likes [1, 1, -1, 1], the only -1 is at channel axe.
            shape = [1 if axis != channel_axis else -1 for axis in range(tensor.ndim)]
            scale, offset = scales.view(shape), offsets.view(shape)

            tensor = ppq_tensor_round((tensor / scale), rounding) + offset
            tensor = torch.clamp(tensor, quant_min, quant_max)
            tensor = (tensor - offset) * scale
            return tensor
        else:

            quantized = CUDA.LinearQuantize_C(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                channel_axis=channel_axis,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value,
            )
            return quantized

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        return grad_outputs[0], None, None, None, None, None, None, None, None, None


class TensorwiseDynamicLinearQuantImpl(Function):
    r"""Torch Tensorwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will use ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, config: TensorQuantizationConfig
    ) -> torch.Tensor:
        # solve scale and offset at first.
        scales, offsets = minmax_to_scale_offset(
            tensor.min().item(), tensor.max().item(), config=config
        )
        print(scales, offsets)
        # quantization function, pytorch implementation
        tensor = ppq_tensor_round((tensor / scales), config.rounding) + offsets
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
        tensor = (tensor - offsets) * scales
        return tensor

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        return grad_outputs[0], None


class ChannelwiseDynamicLinearQuantImpl(Function):
    r"""Torch Channelwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, config: TensorQuantizationConfig
    ) -> torch.Tensor:
        channelwise_view = tensor.transpose(dim0=0, dim1=config.channel_axis).unsqueeze(
            -1
        )
        channelwise_view = torch.flatten(channelwise_view, start_dim=1)

        scales, offsets = [], []
        for _min, _max in zip(
            channelwise_view.min(dim=1)[0].tolist(),
            channelwise_view.max(dim=1)[0].tolist(),
        ):
            s, o = minmax_to_scale_offset(_min, _max, config)
            scales.append(s)
            offsets.append(o)

        scales = torch.tensor(scales, dtype=torch.float32, device=tensor.device)
        offsets = torch.tensor(offsets, dtype=torch.float32, device=tensor.device)

        # generate a shape that likes [1, 1, -1, 1], the only -1 is at channel axe.
        shape = [
            1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)
        ]
        scales, offsets = scales.view(shape), offsets.view(shape)

        tensor = ppq_tensor_round((tensor / scales), config.rounding) + offsets
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
        tensor = (tensor - offsets) * scales
        return tensor

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        return grad_outputs[0], None


def dynamic_linear_quant(
    tensor: torch.Tensor, config: TensorQuantizationConfig
) -> torch.Tensor:
    """
    Dynamic Linear Quantization Function(PPQ 动态量化函数).

    When calling this method, we firstly solve a scale & offset setting
    by min-max observer.

    Then we applies ordinary Linear Quantization Function with solved setting.

    If there is a pre-defined scale & offset within given config,
    they will be dropped without warning.

    动态量化函数将在执行量化之前统计出 tensor 的 min - max, 而后计算出 scale & offset 并完成量化

    此时 TQC 中的 scale 与 offset 将被忽略
    """
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        qtensor = ChannelwiseDynamicLinearQuantImpl.apply(tensor, config)
    else:
        qtensor = TensorwiseDynamicLinearQuantImpl.apply(tensor, config)
    assert isinstance(qtensor, torch.Tensor)
    return qtensor


def linear_fake_quant(
    tensor: torch.Tensor, config: TensorQuantizationConfig
) -> torch.Tensor:
    """PPQ 核心量化函数，没啥好说的了吧，这个玩意既做 quant 也做 dequant"""
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        qtensor = ChannelwiseLinearQuantImpl.apply(
            tensor,
            config.scale,
            config.offset,
            config.channel_axis,
            config.quant_min,
            config.quant_max,
            config.rounding,
        )
    else:
        qtensor = TensorwiseLinearQuantImpl.apply(
            tensor,
            config.scale,
            config.offset,
            config.quant_min,
            config.quant_max,
            config.rounding,
        )
    assert isinstance(qtensor, torch.Tensor)
    return qtensor


def linear_quant_toint(
    tensor: torch.Tensor, config: TensorQuantizationConfig
) -> torch.Tensor:
    """PPQ 核心量化函数，没啥好说的了吧，这个玩意只做 quant 不做 dequant"""
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError("Critical Quantization Error! Non-linear config detected.")
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        shape = [
            1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)
        ]
        scale, offset = config.scale.view(shape), config.offset.view(shape)
        tensor = ppq_tensor_round((tensor / scale), config.rounding) + offset
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
    else:  # config.policy.has_property(QuantizationProperty.PER_TENSOR)
        tensor = (
            ppq_tensor_round((tensor / config.scale), config.rounding) + config.offset
        )
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
    if config.num_of_bits <= 8:
        if config.num_of_bits < 8:
            warning("bits < 8 is still experimental")
        if config.policy.has_property(QuantizationProperty.SYMMETRIC):
            return tensor.type(dtype=torch.int8)
        else:  # config.policy.has_property(QuantizationProperty.ASYMMETRIC)
            return tensor.type(dtype=torch.uint8)
    else:  # config.num_of_bits > 8
        return tensor.type(dtype=torch.int32)
