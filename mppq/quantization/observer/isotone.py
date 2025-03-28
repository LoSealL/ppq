from typing import List

import torch

from mppq.common import OBSERVER_ISOTONE_OBSERVER_AXIS
from mppq.ir.base.graph import Variable
from mppq.logger import warning
from mppq.quant import (
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
)
from mppq.quantization.observer.range import minmax_to_scale_offset

from .base import OBSERVER_TABLE, BaseTensorObserver


@OBSERVER_TABLE.register("isotone")
class TorchIsotoneObserver(BaseTensorObserver):
    """For softmax or sigmoid activations, usually we just need
    argmax(softmax(x)) == argmax(softmax(quant(x)))

    Inspired by this Property, Isotone Observer is designed to provide an
    order-preserving calibration method, which cares only about argmax(x) [or argmin(x)]

    To keep argmax(x) == argmax(quant(x)), we only need to
    distinguish the largest element and the second largert element with quantization

    let L1 represents the largest element of x,
    while L2 represents the second largest.

    For symmetric quantization, We want

        1. L1 - L2 > scale
        2. round(L2 / scale) <= (quant_max - .5)
    """

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        self._cache: List[torch.Tensor] = []
        if OBSERVER_ISOTONE_OBSERVER_AXIS not in quant_cfg.detail:
            warning(
                "Initializing TorchIsotoneObserver with implicit axis"
                " is not recommended."
            )
            self.axis = -1
        else:
            self.axis = quant_cfg.detail[OBSERVER_ISOTONE_OBSERVER_AXIS]
        self.s_candidates = None

    def observe(self, value: torch.Tensor):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        assert isinstance(
            value, torch.Tensor
        ), "IsotoneObserver can only deal with torch Tensor values"
        assert (
            value.numel() > 0
        ), f"You are observing an empty tensor({self._watch_on.name})."

        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            # flatten value as [-1, num_of_elements in isotone axis]
            if value.ndim > 1:
                value = value.transpose(dim0=self.axis, dim1=-1)
                value = value.flatten(start_dim=0, end_dim=-2)
            value, _ = torch.topk(value, k=2, dim=self.axis, largest=True, sorted=True)
            if value.ndim <= 1:
                value = value.unsqueeze(0)
            self._cache.append(value)
        else:
            raise TypeError(
                "Isotone Observer only work with per-tensor quantize policy."
            )

    def render_quantization_config(self):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        device = self._cache[-1].device
        collected = torch.cat(self._cache, dim=0)
        collected = collected.cpu().numpy()
        s_candidates = []

        for l1, l2 in collected:
            if self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRIC):
                l1, l2 = abs(l1), abs(l2)

            scale_min = max(l2 / (self._quant_cfg.quant_max - 0.51), 0)
            scale_max = 2 * (l1 - max(l2, 0))
            if scale_max > scale_min and l1 > 0:
                s_candidates.append((scale_min, "min"))
                s_candidates.append((scale_max, "max"))

        if len(s_candidates) <= 0:
            max_val = collected[-1, 0]
            # fall back to min-max calibration
            scale, offset = minmax_to_scale_offset(
                min_val=0, max_val=max_val, config=self._quant_cfg
            )
            self._quant_cfg.scale = torch.tensor(
                [scale], dtype=torch.float32, device=device
            ).squeeze(0)
            self._quant_cfg.offset = torch.tensor(
                [offset], dtype=torch.float32, device=device
            ).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
            warning(
                "There is no way to make clear classification on "
                f"{self._watch_on.name} under int8 quantization."
            )
            return

        s_candidates = sorted(s_candidates)
        best_satisfied, best_scale, satisfied = 0, 0, 0
        for s_candidate, s_type in s_candidates:
            if s_type == "min":
                satisfied += 1
            if s_type == "max":
                satisfied -= 1

            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_scale = s_candidate

        self._quant_cfg.scale = torch.tensor(
            [best_scale], dtype=torch.float32, device=device
        ).squeeze(0)
        self._quant_cfg.offset = torch.tensor(
            [0], dtype=torch.float32, device=device
        ).squeeze(0)
        self._quant_cfg.state = QuantizationStates.ACTIVATED
        self.s_candidates = s_candidates
