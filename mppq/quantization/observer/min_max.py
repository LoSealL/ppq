from typing import List

import torch

from mppq.ir.base.opdef import Variable
from mppq.logger import nest
from mppq.quant import (
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
)
from mppq.quantization.observer.base import OBSERVER_TABLE, BaseTensorObserver
from mppq.utils.qfunction.linear import minmax_to_scale_offset


@OBSERVER_TABLE.register("minmax")
class TorchMinMaxObserver(BaseTensorObserver):
    """TorchMinMaxObserver collects min and max value of given tensor."""

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        self._min_val_collector: List[torch.Tensor] = []
        self._max_val_collector: List[torch.Tensor] = []
        self._logger = nest(self.__class__.__name__)

    @torch.no_grad()
    def observe(self, value: torch.Tensor):
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return
        assert isinstance(
            value, torch.Tensor
        ), "TorchMinMaxObserver can only deal with torch Tensor values"
        assert (
            value.numel() > 0
        ), f"You are observing an empty tensor({self._watch_on.name})."

        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            self._min_val_collector.append(value.min().reshape(shape=[1]))
            self._max_val_collector.append(value.max().reshape(shape=[1]))
        elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            channel_axis = self._quant_cfg.channel_axis
            channelwise_view = value.transpose(dim0=0, dim1=channel_axis).unsqueeze(-1)
            channelwise_view = torch.flatten(channelwise_view, start_dim=1)
            self._min_val_collector.append(
                torch.min(channelwise_view, dim=1, keepdim=True)[0]
            )
            self._max_val_collector.append(
                torch.max(channelwise_view, dim=1, keepdim=True)[0]
            )
        else:
            raise TypeError(
                "Min-max Observer only work with per-tensor or per-channel "
                "quantize policy."
            )

    def render_quantization_config(self):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        if len(self._max_val_collector) == 0:
            raise ValueError(
                "Can not render quantization config yet, "
                "Observer data collator is empty. "
                "Invoke observe() function before render config."
            )
        device = self._max_val_collector[-1].device

        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            min_val = torch.min(torch.cat(self._min_val_collector, dim=0)).cpu().item()
            max_val = torch.max(torch.cat(self._max_val_collector, dim=0)).cpu().item()
            scale, offset = minmax_to_scale_offset(
                min_val=min_val, max_val=max_val, config=self._quant_cfg
            )
            self._logger.debug(
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
            min_vals = (
                torch.min(
                    torch.cat(self._min_val_collector, dim=-1), dim=-1, keepdim=False
                )[0]
                .cpu()
                .numpy()
            )
            max_vals = (
                torch.max(
                    torch.cat(self._max_val_collector, dim=-1), dim=-1, keepdim=False
                )[0]
                .cpu()
                .numpy()
            )
            assert len(min_vals) == len(
                max_vals
            ), "Min values and max values should at same length."
            scales, offsets = [], []
            for min_val, max_val in zip(min_vals, max_vals):
                scale, offset = minmax_to_scale_offset(
                    min_val=min_val, max_val=max_val, config=self._quant_cfg
                )
                scales.append(scale)
                offsets.append(offset)
            self._logger.debug(
                f"{self._watch_on.name} observes min_val={min_vals}, "
                f"max_val={max_vals}, scale={scales}, zero point={offsets}"
            )
            # scale, offset here only deployed on cpu
            # we will move them towards target device through RunnableGraph
            self._quant_cfg.scale = torch.tensor(
                scales, dtype=torch.float32, device=device
            )
            self._quant_cfg.offset = torch.tensor(
                offsets, dtype=torch.float32, device=device
            )
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        else:
            raise TypeError(
                "Min-max Observer only work with per-tensor or per-channel "
                "quantize policy."
            )
