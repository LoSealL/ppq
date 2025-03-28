from typing import Tuple

import torch

from mppq.common import (
    OBSERVER_MIN_SCALE,
    OBSERVER_MIN_SCALE_MANUL_OVERRIDE,
    OBSERVER_MSE_COMPUTE_INTERVAL,
    OBSERVER_MSE_HIST_BINS,
)
from mppq.defs import ppq_quant_param_computing_function
from mppq.ffi import CUDA, ENABLE_CUDA_KERNEL
from mppq.ir.base.opdef import Variable
from mppq.quant import QuantizationProperty, TensorQuantizationConfig
from mppq.quantization.observer.range import minmax_to_scale_offset

from .base import OBSERVER_TABLE
from .hist import TorchHistObserver


@OBSERVER_TABLE.register("mse")
class TorchMSEObserver(TorchHistObserver):
    """Histogram accelerated MSE Observer, inspired by LightGBM This observer
    will collect data in histogram firstly, all mse computing will directly use
    histogram rather than data itself.

    Time complexity::

        O(Iteration * Num_of_Batch * Length(Data)) -> O(Iteration * Length(histogram))

    Space complexity::

        O(Num_of_Batch * Length(Data)) -> O(Iteration * Length(histogram))

    Args:
        TorchHistObserver ([type]): [description]
    """

    def __init__(
        self,
        watch_on: Variable,
        quant_cfg: TensorQuantizationConfig,
        bins: int = OBSERVER_MSE_HIST_BINS,
    ):
        super().__init__(watch_on, quant_cfg)
        self._hist_bins = bins

    def compute_mse_loss(self, histogram: list, start: int, step: int, end: int):
        if ENABLE_CUDA_KERNEL.USING_CUDA_KERNEL:
            return CUDA.compute_mse_loss(
                histogram=histogram, start=start, step=step, end=end
            )
        else:
            # 如果你觉得 mse 太慢，想办法加速这段代码就可以了
            # 求解 mse 时，我们假设每一个 bin 里面的数据都是均匀分布的
            # 我们需要给一个直方图，并用 start, end, step 给出量化表示的范围
            # losses = [0 for _ in histogram]  debug
            num_of_elements = sum(histogram)
            loss = 0
            for idx, bin in enumerate(histogram):
                if idx < start:
                    # 如果所选的 bin 已经超出了起点，那从 bin 的中心到起点的距离即
                    # ((idx 到 起点的距离) + 0.5)
                    # 注意 hist 统计时是直接取 floor 的，因此会在这里额外 - 1
                    error = (start - idx - 1) + 0.5
                elif idx > end:
                    # 注意 hist 统计时是直接取 floor 的
                    error = (idx - end) + 0.5
                else:
                    # 分别计算左右两侧的 err
                    l_idx = (idx - start) % step
                    r_idx = step - l_idx - 1
                    if l_idx == r_idx:
                        error = l_idx + 0.25
                    else:
                        l_err = l_idx + 0.5
                        r_err = r_idx + 0.5
                        error = min(l_err, r_err)
                loss += (bin * error * error) / num_of_elements
                # losses[idx] = bin * error * error
            return loss

    @ppq_quant_param_computing_function
    def hist_to_scale_offset(  # type: ignore
        self,
        histogram: torch.Tensor,
        hist_bins: int,
        hist_scale: float,
        config: TensorQuantizationConfig,
        scale_threshold: float = OBSERVER_MIN_SCALE,
    ) -> Tuple[float, int]:
        if OBSERVER_MIN_SCALE_MANUL_OVERRIDE in config.detail:
            scale_threshold = config.detail[OBSERVER_MIN_SCALE_MANUL_OVERRIDE]
        hist = histogram.tolist()
        num_of_quant_levels = (
            self._quant_cfg.quant_max - self._quant_cfg.quant_min
        ) + 1

        losses = []
        if config.policy.has_property(
            QuantizationProperty.ASYMMETRIC
        ) and config.policy.has_property(QuantizationProperty.PER_TENSOR):
            # at least we can have a min-max result
            step = hist_bins // num_of_quant_levels + 1
            loss = self.compute_mse_loss(
                histogram=hist, start=0, step=step, end=num_of_quant_levels * step
            )
            losses.append({"mse": loss, "start": 0, "end": num_of_quant_levels * step})

            for start in range(0, hist_bins, OBSERVER_MSE_COMPUTE_INTERVAL):
                if (start * hist_scale) + self._min > 0:
                    break  # start can not > 0

                for step in range(1, hist_bins // num_of_quant_levels + 1):
                    end = start + num_of_quant_levels * step
                    if end > (hist_bins + num_of_quant_levels):
                        break
                    loss = self.compute_mse_loss(
                        histogram=hist, start=start, step=step, end=end
                    )
                    losses.append({"mse": loss, "start": start, "end": end})

            best_policy = sorted(losses, key=lambda x: x["mse"])[0]
            best_start = best_policy["start"]
            best_end = best_policy["end"]

            # translate start & end to scale & offset.
            range_min, range_max = (best_start * hist_scale) + self._min, (
                best_end * hist_scale
            ) + self._min
            scale, offset = minmax_to_scale_offset(
                range_min, range_max, config, scale_threshold
            )
            return scale, int(offset)

        elif config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError(
                "Torch Mse observer do not support PER_CHANNEL policy now, please wait."
            )

        elif config.policy.has_property(
            QuantizationProperty.SYMMETRIC
        ) and config.policy.has_property(QuantizationProperty.PER_TENSOR):
            # at least we can have a min-max result
            step = hist_bins // num_of_quant_levels + 1
            loss = self.compute_mse_loss(
                histogram=hist, start=0, step=step, end=num_of_quant_levels * step
            )
            losses.append({"mse": loss, "end": num_of_quant_levels * step})

            for step in range(1, hist_bins // num_of_quant_levels + 1):
                end = num_of_quant_levels * step
                if end > (hist_bins + num_of_quant_levels):
                    break
                loss = self.compute_mse_loss(
                    histogram=hist, start=0, step=step, end=end
                )
                losses.append({"mse": loss, "end": end})

            best_policy = sorted(losses, key=lambda x: x["mse"])[0]
            best_end = best_policy["end"]

            # translate start & end to scale & offset.
            range_min, range_max = -(best_end * hist_scale), (best_end * hist_scale)
            scale, offset = minmax_to_scale_offset(
                range_min, range_max, config, scale_threshold
            )
            return scale, int(offset)

        raise RuntimeError("Oops, there might be some mistakes.")
