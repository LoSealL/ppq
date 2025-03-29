import inspect
import math
from numbers import Number
from typing import List

import pytest
import torch

from mppq.ffi import CUDA, CUDA_COMPLIER, ENABLE_CUDA_KERNEL
from mppq.quant import RoundingPolicy
from mppq.quantization.measure.norm import torch_snr_error
from mppq.utils.round import ppq_tensor_round


# pylint: disable=missing-function-docstring
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_cuda_compile():
    with ENABLE_CUDA_KERNEL():
        dir(CUDA_COMPLIER.extension)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize(
    "size",
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [5, 12, 13, 4],
        [1, 7, 15, 41],
        [50, 120, 130, 4],
        [12, 74, 15, 411],
        [50, 7, 130, 1],
        [12, 4, 15, 3],
        [5011, 7, 7, 1],
        [122552, 1, 10, 4],
        [10, 10, 124, 47],
        [19, 42, 150, 3],
    ],
)
def test_tensorwise_linear_quantize(size: List[int]):
    with ENABLE_CUDA_KERNEL():
        t = torch.rand(size=size).cuda() * 32
        s = torch.rand(size=[1]).cuda()
        o = torch.randint(low=0, high=255, size=[1]).float().cuda()

        # torch quantize
        qt = ppq_tensor_round(t / s, RoundingPolicy.ROUND_HALF_EVEN) + o
        qt = qt.clip(0, 255)
        qt = (qt - o) * s

        cuda_qt = CUDA.LinearQuantize_T(
            t, s, o, 0, 255, RoundingPolicy.ROUND_HALF_EVEN.value
        )

        diff = qt - cuda_qt
        assert diff.abs().max() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize(
    ("size", "c"),
    [
        ([1, 1, 1, 1], 1),
        ([1, 1, 1, 1], 1),
        ([5, 12, 13, 4], 1),
        ([1, 7, 15, 41], 1),
        ([50, 120, 130, 4], 1),
        ([12, 74, 15, 411], 1),
        ([50, 7, 130, 1], 1),
        ([12, 4, 15, 3], 1),
        ([5011, 7, 7, 1], 0),
        ([122552, 1, 10, 4], 0),
        ([10, 10, 124, 47], 3),
        ([19, 42, 150, 3], 3),
    ],
)
def test_channelwise_linear_quantize(size: List[int], c: int):
    with ENABLE_CUDA_KERNEL():
        nc = size[c]
        t = torch.rand(size=size).cuda() * 32
        s = torch.rand(size=[nc]).cuda()
        o = torch.randint(low=0, high=255, size=[nc]).float().cuda()

        shape = [1 if axis != c else -1 for axis in range(t.ndim)]
        s, o = s.view(shape), o.view(shape)
        qt = ppq_tensor_round(t / s, policy=RoundingPolicy.ROUND_HALF_EVEN) + o
        qt = qt.clip(0, 255)
        qt = (qt - o) * s

        cuda_qt = CUDA.LinearQuantize_C(
            t, s, o, c, 0, 255, RoundingPolicy.ROUND_HALF_EVEN.value
        )

        diff = qt - cuda_qt
        assert diff.abs().max() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize(
    "size",
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [5, 12, 13, 4],
        [1, 7, 15, 41],
        [50, 120, 130, 4],
        [12, 74, 15, 411],
        [50, 7, 130, 1],
        [12, 4, 15, 3],
        [5011, 7, 7, 1],
        [122552, 1, 10, 4],
        [10, 10, 124, 47],
        [19, 42, 150, 3],
    ],
)
def test_tensorwise_linear_quantize_backward(size: List[int]):
    def ref_grad_func(
        value: torch.Tensor,
        dy: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
    ):

        qt = ppq_tensor_round(value / scale, policy=RoundingPolicy.ROUND_HALF_EVEN)
        qt += offset
        clipped_qt = qt.clip(0, 255)

        dx = torch.where(clipped_qt != qt, torch.zeros_like(dy), dy)
        ds = torch.where(
            clipped_qt == qt,
            (((qt - offset) * scale) - value) * dy / scale,
            torch.zeros_like(dy),
        )
        ds += torch.where(qt > 255, (255 - offset) * dy, torch.zeros_like(dy))
        ds += torch.where(qt < 0, (0 - offset) * dy, torch.zeros_like(dy))
        ds = ds.sum() / math.sqrt(value.numel() * (255 - 0))
        return dx, ds

    with ENABLE_CUDA_KERNEL():
        t = torch.rand(size=size).cuda() * 50
        s = torch.rand(size=[1]).cuda()
        o = torch.randint(low=0, high=255, size=[1]).float().cuda()
        dy = torch.rand_like(t)

        cuda_grad_x, cuda_grad_s = CUDA.LinearQuantize_T_B(
            t, s, o, dy, 0, 255, RoundingPolicy.ROUND_HALF_EVEN.value
        )

        ref_grad_x, ref_grad_s = ref_grad_func(t, dy, s, o)
        diff = ref_grad_x.flatten() - cuda_grad_x.flatten()
        assert diff.abs().max() == 0

        snr = torch_snr_error(
            cuda_grad_s.reshape([1, -1]), ref_grad_s.reshape([1, -1])
        ).item()
        if snr > 1e-3:
            assert cuda_grad_s.item() <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize(
    ("size", "c"),
    [
        ([1, 1, 1, 1], 1),
        ([1, 1, 1, 1], 1),
        ([5, 12, 13, 4], 1),
        ([1, 7, 15, 41], 1),
        ([50, 120, 130, 4], 1),
        ([12, 74, 15, 411], 1),
        ([50, 7, 130, 1], 1),
        ([12, 4, 15, 3], 1),
        ([5011, 7, 7, 1], 0),
        ([122552, 1, 10, 4], 0),
        ([10, 10, 124, 47], 3),
        ([19, 42, 150, 3], 3),
    ],
)
def test_channelwise_linear_quantize_backward(size: List[int], c: int):
    def ref_grad_func(
        value: torch.Tensor,
        dy: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
    ):
        shape = [1 if axis != c else -1 for axis in range(t.ndim)]
        scale = scale.view(shape)
        offset = offset.view(shape)

        qt = ppq_tensor_round(value / scale, policy=RoundingPolicy.ROUND_HALF_EVEN)
        qt += offset
        clipped_qt = qt.clip(0, 255)

        dx = torch.where(clipped_qt != qt, torch.zeros_like(dy), dy)
        ds = torch.where(
            clipped_qt == qt,
            (((qt - offset) * scale) - value) * dy / scale,
            torch.zeros_like(dy),
        )
        ds += torch.where(qt > 255, (255 - offset) * dy, torch.zeros_like(dy))
        ds += torch.where(qt < 0, (0 - offset) * dy, torch.zeros_like(dy))
        ds = ds.transpose(0, c).flatten(1).sum(dim=-1) / math.sqrt(value.numel() * 255)
        return dx, ds

    with ENABLE_CUDA_KERNEL():
        nc = size[c]
        t = torch.rand(size=size).cuda() * 32
        s = torch.rand(size=[nc]).cuda()
        o = torch.randint(low=0, high=255, size=[nc]).float().cuda()
        dy = torch.rand_like(t)

        cuda_grad_x, cuda_grad_s = CUDA.LinearQuantize_C_B(
            t, s, o, dy, 0, 255, c, RoundingPolicy.ROUND_HALF_EVEN.value
        )

        ref_grad_x, ref_grad_s = ref_grad_func(t, dy, s, o)
        diff = ref_grad_x.flatten() - cuda_grad_x.flatten()
        assert diff.abs().max() == 0

        snr = torch_snr_error(
            cuda_grad_s.reshape([1, -1]), ref_grad_s.reshape([1, -1])
        ).item()
        if snr > 1e-5:
            assert cuda_grad_s.item() <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize("func", [CUDA.Histogram_T, CUDA.Histogram_Asymmetric_T])
def test_histogram(func):
    with ENABLE_CUDA_KERNEL():
        t = torch.rand(size=[128, 3, 224, 224]).to("cuda")
        s = 0.01

        # torch hist
        hist = torch.histc(torch.abs(t), bins=50, min=0, max=0.5)

        cuda_hist = torch.zeros(size=[50]).to("cuda").int()
        if func is CUDA.Histogram_T:
            cuda_hist = func(t, cuda_hist, s)
        elif func is CUDA.Histogram_Asymmetric_T:
            cuda_hist = func(0, 0.5, t, cuda_hist, clip_outliers=True)
        assert torch.abs(hist - cuda_hist).max().item() < 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_quantile():
    percentile = 0.999
    with ENABLE_CUDA_KERNEL():
        t = torch.randn(128, 3, 224, 224).to("cuda")
        numel = t.numel()
        min_idx, max_idx = int(numel * (1 - percentile)), int(numel * percentile)
        # torch.kthvalue needs index from 1 to numel ...
        min_idx = max(0, min_idx) + 1
        max_idx = min(max_idx, numel - 1) + 1
        _min = torch.kthvalue(t.flatten(), k=min_idx, dim=0)[0].view(1, -1)
        _max = torch.kthvalue(t.flatten(), k=max_idx, dim=0)[0].view(1, -1)
        ref_stat = torch.cat([_max, _min], dim=-1)

        cuda_stat = CUDA.Quantile(t, percentile).view(1, -1)
        assert torch.abs(ref_stat - cuda_stat).max() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_compute_mse_loss():
    def ref_mse_loss(histogram: list, start: int, step: int, end: int):
        num_of_elements = sum(histogram)
        loss = 0
        for idx, hist in enumerate(histogram):
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
            loss += (hist * error * error) / num_of_elements
        return loss

    with ENABLE_CUDA_KERNEL():
        histograms = torch.randint(0, 10000, size=[50]).tolist()
        start, step, end = 10, 2, 20

        ref_loss = ref_mse_loss(histograms, start, step, end)
        cpu_loss = CUDA.compute_mse_loss(histograms, start, step, end)
        assert abs(ref_loss - cpu_loss) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize(
    "func",
    [
        CUDA.Histogram_C,
        CUDA.RoundingLoss_LC,
        CUDA.RoundingLoss_LC_B,
        CUDA.RoundingLoss_LT,
        CUDA.RoundingLoss_LT_B,
        CUDA.FloatingQuantize_C,
        CUDA.FloatingQuantize_C_B,
        CUDA.FloatingQuantize_T,
        CUDA.FloatingQuantize_T_B,
    ],
)
@pytest.mark.parametrize(
    ("size", "c"),
    [
        ([1, 1, 1, 1], 1),
        ([1, 1, 1, 1], 1),
        ([5, 12, 13, 4], 1),
        ([1, 7, 15, 41], 1),
        ([50, 120, 130, 4], 1),
        ([12, 74, 15, 411], 1),
        ([50, 7, 130, 1], 1),
        ([12, 4, 15, 3], 1),
        ([5011, 7, 7, 1], 0),
        ([122552, 1, 10, 4], 0),
        ([10, 10, 124, 47], 3),
        ([19, 42, 150, 3], 3),
    ],
)
def test_fuzzy_func_call(func, size: List[int], c: int):

    def args_from_signature(signature):
        signature = inspect.signature(func)
        nc = size[c]
        for k, v in signature.parameters.items():
            if k in ("tensor", "dy") and v.annotation == torch.Tensor:
                yield k, torch.randn(size).cuda() * 32
            elif k in ("histogram") and v.annotation == torch.Tensor:
                yield k, torch.randint(0, 256, size=[nc]).to(torch.int32).cuda()
            elif v.annotation == torch.Tensor:
                yield k, torch.randn([nc]).cuda()
            elif k == "channel_axis":
                yield k, c
            elif v.default is signature.empty and issubclass(v.annotation, Number):
                yield k, v.annotation()

    with ENABLE_CUDA_KERNEL():
        kw = {k: v for k, v in args_from_signature(func)}
        func(**kw)
