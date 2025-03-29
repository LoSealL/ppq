"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Default quantizer for example.
"""

from mppq.common import COMPUTING_OP
from mppq.ir.base.graph import Operation
from mppq.quant import (
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TargetPrecision,
    TensorQuantizationConfig,
)
from mppq.quantization.optim.base import (
    OPTIM_ALGORITHMS,
    QuantizationOptimizationPipeline,
)
from mppq.quantizer.base import QUANTIZER, BaseQuantizer


def create_default_quant_config(
    op: Operation,
    num_of_bits: int = 8,
    quant_min: int = 0,
    quant_max: int = 255,
    observer_algorithm: str = "percentile",
    policy: QuantizationPolicy = QuantizationPolicy(
        QuantizationProperty.PER_TENSOR
        | QuantizationProperty.LINEAR
        | QuantizationProperty.ASYMMETRIC
    ),
    rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN,
    exponent_bits: int = 0,
) -> OperationQuantizationConfig:
    r"""为你的算子创建一个默认量化信息

    对于一个 Onnx 算子而言，它总是会有几个输入和输出 Variable
    你需要为每一个相连的 Variable 初始化量化信息 TensorQuantConfig

    这个函数就是用来帮你初始化这些信息的。

    一个麻烦的问题是：

        对于很多 onnx 算子而言，他们的部分输入都是不需要量化的:

        如 Clip 算子的三个输入 value, min, max, 大部分框架不要求量化 min, max
        如 Reshape 算子的两个输入 value, shape, 其中 shape 不能够被量化

        PPQ 的算子接线器中记录了这些信息

        算子接线器中记录了所有标准 onnx 的默认量化策略
        该函数将使用预定义的算子量化策略初始化量化信息

    你可以在 Quantizer 中对默认量化策略进行进一步修改

    Create a default quantization configuration for given op.
    For each onnx op, there will be some input and output variables.

    You are required to create tensor quantization config for every
    input and output variables.

    This function is designed for creating a default quantization config for you.

        The created OQC(Op Quantization Config) is based on OpSocket.

        In fact, there are some rules or templates when creating the OQC:
        For Clip Op which has 3 input variable, namely value, min and max
            most framework does not require a quantization config for min and max.
        For Reshape Op which has 2 input variable, namely value and shape
            the input shape can never be quantized.

    Those rules are pre-defined within OpSocket, thus ppq will create
    OQC based on underlying OpSocket of your Op.

    After the default OQC got created, you can overwrite its state in quantizer.
    """
    assert isinstance(
        op, Operation
    ), f"Can only initialize OQC for PPQ.IR.Operation, however {type(op)} was given."
    assert isinstance(
        policy, QuantizationPolicy
    ), "Can not create quantization config - Quantization Policy Type Error."
    assert isinstance(
        rounding, RoundingPolicy
    ), "Can not create quantization config - Rounding Policy Type Error."

    socket = op.socket
    input_cfgs, output_cfgs = [], []
    for index in range(op.num_of_input):
        state = QuantizationStates.INITIAL
        # for those unexpected inputs and outputs
        # ppq just initialize them as normal variable.
        if index < len(socket.in_plat):
            target_plat = socket.in_plat[index]
            if target_plat == TargetPrecision.FP32:
                state = QuantizationStates.FP32
            if target_plat == TargetPrecision.SOI:
                state = QuantizationStates.FP32
        input_cfgs.append(
            TensorQuantizationConfig(
                policy=policy,
                rounding=rounding,
                num_of_bits=num_of_bits,
                scale=None,
                offset=None,
                exponent_bits=exponent_bits,
                quant_min=quant_min,
                quant_max=quant_max,
                observer_algorithm=observer_algorithm,
                state=state,
            )
        )

    for index in range(op.num_of_output):
        state = QuantizationStates.INITIAL
        # for those unexpected inputs and outputs
        # ppq just initialize them as normal variable.
        if index < len(socket.out_plat):
            target_plat = socket.out_plat[index]
            if target_plat == TargetPrecision.FP32:
                state = QuantizationStates.FP32
            if target_plat == TargetPrecision.SOI:
                state = QuantizationStates.FP32
        output_cfgs.append(
            TensorQuantizationConfig(
                policy=policy,
                rounding=rounding,
                num_of_bits=num_of_bits,
                scale=None,
                offset=None,
                exponent_bits=exponent_bits,
                quant_min=quant_min,
                quant_max=quant_max,
                observer_algorithm=observer_algorithm,
                state=state,
            )
        )

    return OperationQuantizationConfig(input_cfgs, output_cfgs)


@QUANTIZER.register("default")
class DefaultQuantizer(BaseQuantizer):
    r"""默认量化器，用于示例。"""

    @property
    def quant_operation_types(self):
        return set(COMPUTING_OP)

    def init_quantize_config(self, operation):
        assert operation.type in self.quant_operation_types

        base_qc = create_default_quant_config(operation)
        weight_policy = QuantizationPolicy(
            QuantizationProperty.PER_CHANNEL
            | QuantizationProperty.LINEAR
            | QuantizationProperty.SYMMETRIC
        )
        weight_min, weight_max = -128, 127
        if operation.type in {"Conv", "ConvTranspose", "Gemm", "MatMul"}:
            # base_qc.output_quantization_config[0].state = QuantizationStates.FP32
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel
            # quant-config.
            assert (
                operation.num_of_input > 0
            ), f"Seems you got a {operation.type} layer with no parameters."

            # first parameter must exist, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {"Conv", "ConvTranspose"}:
                if operation.inputs[1].is_parameter:
                    conv_weight_config = base_qc.input_quantization_config[1]
                    conv_weight_config.quant_min = weight_min
                    conv_weight_config.quant_max = weight_max
                    conv_weight_config.policy = weight_policy
                    conv_weight_config.channel_axis = 1
                    if operation.type == "ConvTranspose":
                        conv_weight_config.channel_axis = 0
                    conv_weight_config.observer_algorithm = "minmax"
            # first parameter must exist, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {"Gemm", "MatMul"}:
                if operation.inputs[1].is_parameter:
                    gemm_weight_config = base_qc.input_quantization_config[1]
                    gemm_weight_config.quant_min = weight_min
                    gemm_weight_config.quant_max = weight_max
                    gemm_weight_config.policy = weight_policy
                    gemm_weight_config.channel_axis = 1
                    if operation.type == "Gemm" and operation.attributes.get(
                        "transB", None
                    ):
                        gemm_weight_config.channel_axis = 0
                    gemm_weight_config.observer_algorithm = "minmax"
            if operation.num_of_input > 2:
                bias_config = base_qc.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32
        elif operation.type == "LayerNormalization":
            # Layernorm - gamma and beta need to be FP32
            for qc in base_qc.input_quantization_config[1:]:
                qc.state = QuantizationStates.FP32
        return base_qc

    @property
    def default_prequant_pipeline(self) -> QuantizationOptimizationPipeline:
        return QuantizationOptimizationPipeline(
            [
                OPTIM_ALGORITHMS["ActivationEqualizationPass"](),
                OPTIM_ALGORITHMS["LayerwiseEqualizationPass"](),
                OPTIM_ALGORITHMS["ChannelwiseSplitPass"](),
            ]
        )

    @property
    def default_quant_pipeline(self) -> QuantizationOptimizationPipeline:
        return QuantizationOptimizationPipeline(
            [
                OPTIM_ALGORITHMS["SSDEqualizationPass"](),
                OPTIM_ALGORITHMS["QuantizeFusionPass"](),
                OPTIM_ALGORITHMS["QuantizeSimplifyPass"](),
                OPTIM_ALGORITHMS["ParameterQuantizePass"](),
                OPTIM_ALGORITHMS["IsotoneCalibrationPass"](),
                OPTIM_ALGORITHMS["RuntimeCalibrationPass"](),
                OPTIM_ALGORITHMS["QuantAlignmentPass"](),
                OPTIM_ALGORITHMS["PassiveParameterQuantizePass"](),
                OPTIM_ALGORITHMS["BiasCorrectionPass"](),
                OPTIM_ALGORITHMS["LearnedStepSizePass"](),
                OPTIM_ALGORITHMS["AdaroundPass"](),
                OPTIM_ALGORITHMS["RoundTuningPass"](),
                OPTIM_ALGORITHMS["PassiveParameterQuantizePass"](),
                OPTIM_ALGORITHMS["ParameterBakingPass"](),
            ]
        )
