from typing import Tuple

from mppq.common import OBSERVER_MIN_SCALE, OBSERVER_MIN_SCALE_MANUL_OVERRIDE
from mppq.defs import ppq_quant_param_computing_function
from mppq.logger import warning
from mppq.quant import QuantizationProperty, RoundingPolicy, TensorQuantizationConfig
from mppq.utils.round import ppq_numerical_round, ppq_round_to_power_of_2


@ppq_quant_param_computing_function
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
