import math
from functools import partial
from typing import Dict, List, Literal, Optional, Sequence

import torch

from mppq.executor.base import RuntimeHook
from mppq.ir.base.opdef import Operation
from mppq.logger import warning
from mppq.quantization.measure.cosine import torch_cosine_similarity
from mppq.quantization.measure.norm import torch_mean_square_error, torch_snr_error
from mppq.utils.fetch import batch_random_fetch, tensor_random_fetch


class OutputRecorder(RuntimeHook):
    """Record model output in the runtime hook."""

    def __init__(self, operation: Operation, fetches: int = 4096) -> None:
        self.fetched: Optional[torch.Tensor] = None
        self.fetches = fetches
        super().__init__(operation)

    def post_forward_hook(self, outputs: Sequence[torch.Tensor], **kwargs):
        output_tensor = outputs[0]
        assert isinstance(
            output_tensor, torch.Tensor
        ), "Output of monitoring operation is not a torch.Tensor"
        self.fetched = batch_random_fetch(
            output_tensor, seed=10086, fetches_per_batch=self.fetches
        ).to("cpu")
        return super().post_forward_hook(outputs, **kwargs)

    def pop(self) -> torch.Tensor:
        assert self.fetched is not None
        fetched = self.fetched
        self.fetched = None
        return fetched


class DetailedRecorder(RuntimeHook):
    def __init__(self, operation: Operation, fetches: int = 1024) -> None:
        self.fetches = fetches
        self.i_storage = [[] for _ in range(operation.num_of_input)]
        self.o_storage = [[] for _ in range(operation.num_of_output)]
        super().__init__(operation)

    def pre_forward_hook(self, inputs: Sequence[torch.Tensor], **kwargs):
        for i, v in enumerate(inputs):
            vv = tensor_random_fetch(v, seed=10086, num_of_fetches=self.fetches)
            self.i_storage[i].append(vv.to("cpu"))
        return super().pre_forward_hook(inputs, **kwargs)

    def post_forward_hook(self, outputs: Sequence[torch.Tensor], **kwargs):
        for i, v in enumerate(outputs):
            vv = tensor_random_fetch(v, seed=10086, num_of_fetches=self.fetches)
            self.o_storage[i].append(vv.to("cpu"))
        return super().post_forward_hook(outputs, **kwargs)

    def clear(self):
        self.i_storage = [[] for _ in range(self._hook_to.num_of_input)]
        self.o_storage = [[] for _ in range(self._hook_to.num_of_output)]


class MeasureRecorder:
    """Helper class for collecting data."""

    def __init__(self, measurement: str = "cosine", reduce: str = "mean") -> None:
        self.num_of_elements: int = 0
        self.measure: float = 0
        if reduce not in {"mean", "max"}:
            raise ValueError(
                "PPQ MeasureRecorder Only support reduce by mean or max, "
                f"however {reduce} was given."
            )

        if str(measurement).lower() == "cosine":
            measure_fn = partial(torch_cosine_similarity, reduction=reduce)
        elif str(measurement).lower() == "mse":
            measure_fn = partial(torch_mean_square_error, reduction=reduce)
        elif str(measurement).lower() == "snr":
            measure_fn = partial(torch_snr_error, reduction=reduce)
        else:
            raise ValueError(
                "Unsupported measurement detected. PPQ only support mse, "
                f"snr and consine now, while {measurement} was given."
            )

        self.measure_fn = measure_fn
        self.reduce = reduce

    def update(self, y_pred: torch.Tensor, y_real: torch.Tensor):
        elements = y_pred.shape[0]
        if elements != y_real.shape[0]:
            raise ValueError(
                "Can not update measurement, cause your input data do not share a "
                f"same batchsize. Shape of y_pred {y_pred.shape} - against shape of "
                f"y_real {y_real.shape}"
            )
        result = self.measure_fn(y_pred=y_pred, y_real=y_real).item()

        if self.reduce == "mean":
            self.measure = self.measure * self.num_of_elements + result * elements
            self.num_of_elements += elements
            self.measure /= self.num_of_elements

        if self.reduce == "max":
            self.measure = max(self.measure, result)
            self.num_of_elements += elements


class MeasurePrinter:
    """Helper class for print top-k record."""

    def __init__(
        self,
        data: Dict[str, float],
        measure: str,
        label: str = "Layer",
        k: Optional[int] = None,
        order: Optional[Literal["large_to_small", "small_to_large"]] = "large_to_small",
        percentage: bool = False,
    ) -> None:

        if order not in {"large_to_small", "small_to_large", None}:
            raise ValueError(
                'Parameter "order" can only be "large_to_small" or "small_to_large"'
            )
        self.collection = [(name, value) for name, value in data.items()]
        if order is not None:
            self.collection = sorted(self.collection, key=lambda x: x[1])
            if order == "large_to_small":
                self.collection = self.collection[::-1]
        if k is not None:
            self.collection = self.collection[:k]

        if order is None:
            sorted_collection = sorted(self.collection, key=lambda x: x[1])
            largest_element, smallest_element = (
                sorted_collection[-1][1],
                sorted_collection[0][1],
            )
        elif order == "large_to_small":
            largest_element, smallest_element = (
                self.collection[0][1],
                self.collection[-1][1],
            )
        else:  # order == "small_to_large"
            largest_element, smallest_element = (
                self.collection[-1][1],
                self.collection[0][1],
            )
        self.normalized_by = largest_element - smallest_element
        self.min = smallest_element

        max_name_length = len(label)
        for name, _ in self.collection:
            max_name_length = max(len(name), max_name_length)
        self.max_name_length = max_name_length
        self.measure_str = measure
        self.label = label
        self.percentage = percentage

    def print(self, max_blocks: int = 20, ansi: bool = False):
        """Pretty print of measured data."""

        print(f"{self.label:<{self.max_name_length}} | {self.measure_str}")
        ch = "=" if ansi else "█"  # \u2588
        for name, value in self.collection:
            normalized_value = (value - self.min) / (self.normalized_by + 1e-7)
            if math.isnan(value):
                warning("MeasurePrinter found an NaN value in your data.")
                normalized_value = 0
            num_of_blocks = round(normalized_value * max_blocks)
            # align the text
            num_of_blocks = max(min(max_blocks - 1, num_of_blocks), 1)

            msg = f"{name:<{self.max_name_length}} "
            msg += f"{'|':{ch}<{num_of_blocks}}{'|':>{max_blocks - num_of_blocks}} "
            if self.percentage:
                msg += f"{value * 100:.3f}%"
            else:
                msg += f"{value:.4f}"
            print(msg)
