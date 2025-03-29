from typing import Optional, Tuple

import torch
from torch.ao.quantization.observer import HistogramObserver

from mppq.common import (
    OBSERVER_KL_COMPUTING_DEVICE,
    OBSERVER_KL_HIST_BINS,
    OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE,
    OBSERVER_MIN_SCALE,
    OBSERVER_MIN_SCALE_MANUL_OVERRIDE,
)
from mppq.defs import ppq_quant_param_computing_function
from mppq.ffi import CUDA, ENABLE_CUDA_KERNEL
from mppq.ir.base.opdef import Variable
from mppq.quant import (
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TensorQuantizationConfig,
)
from mppq.quantization.measure.statistic import torch_KL_divergence
from mppq.quantization.observer.base import OBSERVER_TABLE, BaseTensorObserver
from mppq.quantization.observer.min_max import TorchMinMaxObserver
from mppq.utils.round import ppq_round_to_power_of_2


@OBSERVER_TABLE.register("kl")
class TorchHistObserver(TorchMinMaxObserver):
    r"""TorchHistObserver collects histogram of given tensor.

    It is designed for per-tensor quantization or activation quantization.
    """

    def __init__(
        self,
        watch_on: Variable,
        quant_cfg: TensorQuantizationConfig,
        hist_bins: int = OBSERVER_KL_HIST_BINS,
    ):
        self._phase = "Detecting Minmax"
        self._hist: Optional[torch.Tensor] = None
        self._hist_scale: float = 1
        self._min: float = 0
        self._max: float = 0
        if OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE in quant_cfg.detail:
            hist_bins = quant_cfg.detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE]
        self._hist_bins: int = hist_bins
        super().__init__(watch_on, quant_cfg)

    def observe(self, value: torch.Tensor):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        assert (
            value.numel() > 0
        ), f"You are observing an empty tensor({self._watch_on.name})."

        if self._phase == "Detecting Minmax":
            return super().observe(value)  # collect min, max

        elif self._phase == "Collating Hist":
            if self._hist is None:
                self._hist = torch.zeros(
                    size=(self._hist_bins,), dtype=torch.int32, device=value.device
                )

            if self._quant_cfg.policy.has_property(QuantizationProperty.ASYMMETRIC):
                # ASYMMETRICAL Hist
                if ENABLE_CUDA_KERNEL.USING_CUDA_KERNEL and value.is_cuda:
                    CUDA.Histogram_Asymmetric_T(
                        self._min, self._max, tensor=value, histogram=self._hist
                    )
                else:
                    hist = torch.histc(
                        value, self._hist_bins, min=self._min, max=self._max
                    )
                    self._hist += hist.int()

            elif self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRIC):
                # SYMMETRICAL Hist
                if ENABLE_CUDA_KERNEL.USING_CUDA_KERNEL and value.is_cuda:
                    CUDA.Histogram_T(
                        tensor=value, histogram=self._hist, scale=self._hist_scale
                    )
                else:
                    hist = torch.histc(
                        torch.abs(value),
                        self._hist_bins,
                        min=0,
                        max=self._hist_scale * self._hist_bins,
                    )
                    self._hist += hist.int()

            else:
                raise TypeError(
                    "Quantization Property is invalid, "
                    "expect either ASYMMETRICAL or SYMMETRICAL config here."
                )

    @ppq_quant_param_computing_function
    def hist_to_scale_offset(
        self,
        histogram: torch.Tensor,
        hist_bins: int,
        hist_scale: float,
        config: TensorQuantizationConfig,
        computing_device: str = OBSERVER_KL_COMPUTING_DEVICE,
        scale_threshold: float = OBSERVER_MIN_SCALE,
    ) -> Tuple[float, int]:
        """
        PPQ core quant parameter computing method - Histogram to scale & offset

        With a pre-defined histogram,
        this function will automatically search best clip value
        to minimize KL divergence between quantized result and fp32 input.

        only work for per-tensor symmetrical quantization policy for now.
        see also

        https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

        Args:
            histogram (torch.Tensor): histogram records activation's statistics.
            hist_bins (int): how many bins are included in histogram
                (also known as histogram length)
            hist_scale (float): histogram step size. it can be solved by
                histogram.max_val / histogram.bins
            config (TensorQuantizationConfig): quantization config.
            computing_device (str, optional): computing device. Defaults to 'cpu'.

        Raises:
            ValueError: given quantization config is invalid.

        Returns:
            Tuple[float, int]: scale(fp32) and offset(int).
        """
        if config.policy.has_property(QuantizationProperty.ASYMMETRIC):
            raise PermissionError(
                "KL observer is not designed for ASYMMETRICAL quantization"
            )

        if OBSERVER_MIN_SCALE_MANUL_OVERRIDE in config.detail:
            scale_threshold = config.detail[OBSERVER_MIN_SCALE_MANUL_OVERRIDE]

        # move histogram to cpu, speedup computation.
        histogram = histogram.to(computing_device).float()

        # compute symmtrical kl-divergence.
        # Here is a simple example: reference distribution P consisting of 8 bins,
        # we want to quantize into 2 bins:
        # P = [ 1, 0, 2, 3, 5, 3, 1, 7]
        # we merge into 2 bins (8 / 2 = 4 consecutive bins are merged into one bin)
        # [1 + 0 + 2 + 3 , 5 + 3 + 1 + 7] = [6, 16]
        # then proportionally expand back to 8 bins, we preserve empty bins from the
        # original distribution P:
        # Q = [ 6/3, 0, 6/3, 6/3, 16/4, 16/4, 16/4, 16/4] = [ 2, 0, 2, 2, 4, 4, 4, 4]
        # now we should normalize both distributions, after that we can compute
        # KL_divergence
        # P /= sum(P) Q /= sum(Q)
        # result = KL_divergence(P, Q)
        # see also
        # https://github.com/NVIDIA/TensorRT/blob/3835424af081db4dc8cfa3ff3c9f4a8b89844421/tools/pytorch-quantization/pytorch_quantization/calib/histogram.py#L147

        losses, quant_bins = [], 2 ** (config.num_of_bits - 1)

        # following code is curcial, do not remove
        histogram[: int(hist_bins * 0.002)] = 0
        histogram[int(hist_bins * 0.002)] = 1

        hist_sum = torch.sum(histogram)
        for bin_range in range(quant_bins, hist_bins + quant_bins - 1, quant_bins):
            p_hist = torch.zeros(
                size=(bin_range,), dtype=torch.float, device=computing_device
            )
            p_hist[:bin_range].copy_(histogram[:bin_range])
            p_hist[bin_range - 1] += torch.sum(histogram[bin_range:])
            p_hist = p_hist / hist_sum

            expand_ratio = int(bin_range / quant_bins)
            q_hist = histogram[:bin_range].clone()
            q_hist = q_hist.reshape((quant_bins, expand_ratio))
            positive_map = q_hist > 0
            positive_cnt = positive_map.sum(dim=1, keepdim=True)
            positive_cnt[positive_cnt == 0] = 1
            q_hist = torch.div(q_hist.sum(dim=1, keepdim=True), positive_cnt)
            q_hist = q_hist.repeat([1, expand_ratio])
            q_hist = q_hist * positive_map
            q_hist = q_hist / torch.sum(q_hist)
            q_hist = q_hist.flatten()

            losses.append(
                {"kl": torch_KL_divergence(p_hist, q_hist), "bin_range": bin_range}
            )

        best_bin_range = sorted(losses, key=lambda x: x["kl"])[0]["bin_range"]
        scale, offset = (best_bin_range / self._hist_bins) * hist_scale * (
            self._hist_bins / quant_bins
        ), 0

        if scale < scale_threshold:
            self._logger.warning(
                "Numeric instability detected: "
                "ppq find there is a scale value < 1e-7, "
                "which probably cause numeric underflow in further computation."
            )
        scale = max(scale, scale_threshold)

        if config.policy.has_property(QuantizationProperty.POWER_OF_2):
            scale = ppq_round_to_power_of_2(scale, policy=RoundingPolicy.ROUND_HALF_UP)
        return scale, offset

    def render_quantization_config(self):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        if not self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            raise ValueError(
                "Hist observer can only apply with per-tensor quantization config."
            )

        if self._phase == "Detecting Minmax":
            min_val = torch.min(torch.cat(self._min_val_collector, dim=0)).cpu().item()
            max_val = torch.max(torch.cat(self._max_val_collector, dim=0)).cpu().item()
            if self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRIC):
                hist_range = float(max(abs(max_val), abs(min_val)))
            else:
                hist_range = max_val - min_val
            self._min = min_val
            self._max = max_val
            self._hist_scale = hist_range / self._hist_bins
            self._phase = "Collating Hist"
        elif self._phase == "Collating Hist":
            assert self._hist is not None
            scale, offset = self.hist_to_scale_offset(
                histogram=self._hist,
                hist_bins=self._hist_bins,
                hist_scale=self._hist_scale,
                config=self._quant_cfg,
            )
            device = self._hist.device
            self._quant_cfg.scale = torch.tensor(
                [scale], dtype=torch.float32, device=device
            ).squeeze(0)
            self._quant_cfg.offset = torch.tensor(
                [offset], dtype=torch.float32, device=device
            ).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED


@OBSERVER_TABLE.register(name="hist")
class TorchHistogramObserver(BaseTensorObserver):
    """Porting torch original hist observer."""

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        if OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE in quant_cfg.detail:
            hist_bins = quant_cfg.detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE]
        else:
            hist_bins = OBSERVER_KL_HIST_BINS
        if quant_cfg.policy.has_property(QuantizationProperty.ASYMMETRIC):
            dtype = torch.quint8
            qscheme = torch.per_tensor_affine
        else:
            dtype = torch.qint8
            qscheme = torch.per_tensor_symmetric
        self._ob = HistogramObserver(hist_bins, dtype=dtype, qscheme=qscheme)

    @torch.no_grad()
    def observe(self, value: torch.Tensor):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        assert (
            value.numel() > 0
        ), f"You are observing an empty tensor({self._watch_on.name})."

        if self._ob.histogram.device != value.device:
            self._ob.to(value.device)
        self._ob(value)

    def render_quantization_config(self):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        if not self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            raise ValueError(
                "Hist observer can only apply with per-tensor quantization config."
            )

        scales, zero_points = self._ob.calculate_qparams()
        device = self._ob.histogram.device
        self._quant_cfg.scale = scales.to(device=device).squeeze(0)
        self._quant_cfg.offset = zero_points.to(device=device).squeeze(0).float()
        self._quant_cfg.state = QuantizationStates.ACTIVATED
