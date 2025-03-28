from typing import List

import torch

from mppq.common import OBSERVER_PERCENTILE, OBSERVER_PERCENTILE_MANUL_OVERRIDE
from mppq.ffi import CUDA, ENABLE_CUDA_KERNEL
from mppq.ir.base.opdef import Variable
from mppq.logger import debug
from mppq.quant import (
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
)
from mppq.quantization.observer.range import minmax_to_scale_offset

from .base import OBSERVER_TABLE, BaseTensorObserver


@OBSERVER_TABLE.register("percentile")
class TorchPercentileObserver(BaseTensorObserver):
    """TorchPercentileObserver collects percentile data of given tensor.

    It is designed for activation quantization.
    """

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        if OBSERVER_PERCENTILE_MANUL_OVERRIDE not in quant_cfg.detail:
            self._percentile = OBSERVER_PERCENTILE
        else:
            self._percentile = quant_cfg.detail[OBSERVER_PERCENTILE_MANUL_OVERRIDE]
        self._percentile_collector: List[torch.Tensor] = []
        self._percentile_maxs: List[torch.Tensor] = []
        self._percentile_mins: List[torch.Tensor] = []

    @torch.no_grad()
    def observe(self, value: torch.Tensor):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        assert value is not None, (
            "You are observing an Empty Tensor. "
            "(This Error is usually due to you have a wrong Quantizer configuration.)"
        )
        assert (
            value.numel() > 0
        ), f"You are observing an empty tensor({self._watch_on.name})."
        assert isinstance(
            value, torch.Tensor
        ), "TorchMinMaxObserver can only deal with torch Tensor values"

        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            if not ENABLE_CUDA_KERNEL.USING_CUDA_KERNEL or (not value.is_cuda):
                numel = value.numel()

                min_idx, max_idx = int(numel * (1 - self._percentile)), int(
                    numel * (self._percentile)
                )
                # torch.kthvalue needs index from 1 to numel ...
                min_idx = max(0, min_idx) + 1
                max_idx = min(max_idx, numel - 1) + 1
                _min = torch.kthvalue(value.flatten(), k=min_idx, dim=0)[0].view(1, -1)
                _max = torch.kthvalue(value.flatten(), k=max_idx, dim=0)[0].view(1, -1)
                self._percentile_collector.append(torch.cat([_max, _min], dim=-1))
            else:
                self._percentile_collector.append(
                    CUDA.Quantile(value, self._percentile).view(1, -1)
                )
        elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            channel_axis = self._quant_cfg.channel_axis
            channelwise_view = value.transpose(dim0=0, dim1=channel_axis)
            channelwise_view = torch.flatten(channelwise_view, start_dim=1)
            self._percentile_mins.append(
                -torch.quantile(
                    -channelwise_view, q=self._percentile, dim=1, keepdim=True
                )[0]
            )
            self._percentile_maxs.append(
                torch.quantile(
                    channelwise_view, q=self._percentile, dim=1, keepdim=True
                )[0]
            )
        else:
            raise TypeError(
                "Percentile Observer only work with per-tensor or per-channel "
                "quantize policy."
            )

    def render_quantization_config(self):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            if len(self._percentile_collector) == 0:
                raise ValueError(
                    "Can not render quantization config yet, "
                    "Observer data collator is empty. "
                    "Invoke observe() function before render config."
                )
            device = self._percentile_collector[-1].device
            percentile_collector = (
                torch.cat(self._percentile_collector, dim=0).float().mean(dim=0).cpu()
            )
            min_val = percentile_collector[1].item()
            max_val = percentile_collector[0].item()
            scale, offset = minmax_to_scale_offset(
                min_val=min_val, max_val=max_val, config=self._quant_cfg
            )
            debug(
                f"{self._watch_on.name} observes min_val={min_val}, max_val={max_val}, "
                f"scale={scale}, zero point={offset}"
            )

            self._quant_cfg.scale = torch.tensor(
                [scale], dtype=torch.float32, device=device
            ).squeeze(0)
            self._quant_cfg.offset = torch.tensor(
                [offset], dtype=torch.float32, device=device
            ).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            if len(self._percentile_maxs) == 0:
                raise ValueError(
                    "Can not render quantization config yet, "
                    "Observer data collator is empty. "
                    "Invoke observe() function before render config."
                )
            device = self._percentile_maxs[-1].device

            min_vals = torch.mean(
                torch.cat(self._percentile_mins, dim=-1), dim=-1, keepdim=False
            )
            max_vals = torch.mean(
                torch.cat(self._percentile_maxs, dim=-1), dim=-1, keepdim=False
            )

            min_vals = min_vals.cpu()
            max_vals = max_vals.cpu()

            assert len(min_vals) == len(
                max_vals
            ), "Min values and max values should at same length."
            scales, offsets = [], []
            for min_val, max_val in zip(min_vals, max_vals):
                scale, offset = minmax_to_scale_offset(
                    min_val=float(min_val),
                    max_val=float(max_val),
                    config=self._quant_cfg,
                )
                scales.append(scale)
                offsets.append(offset)

            self._quant_cfg.scale = torch.tensor(
                scales, dtype=torch.float32, device=device
            )
            self._quant_cfg.offset = torch.tensor(
                offsets, dtype=torch.int32, device=device
            )
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        else:
            raise TypeError(
                "Percentile Observer only work with per-tensor or per-channel "
                "quantize policy."
            )
