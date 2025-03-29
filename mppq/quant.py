"""PPQ Core Data Structure Abstraction PPQ 核心量化结构抽象.

You are not allowed to modify this 请勿修改此文件
"""

import time  # for hash generation
from copy import deepcopy
from enum import Enum, IntEnum
from typing import Any, Dict, Iterator, List, Optional

import torch

from mppq.common import EXPORT_OVERLAPPED_CONFIG
from mppq.storage import Serializable


class QuantVisibility(Enum):
    r"""量化参数可见性

    可见性用来控制量化参数是否被导出。不可见的参数不会被导出到最终模型里。
    """

    FORCE_EXPORT = 1
    r"""强制导出，无论该参数是否被正确激活。"""

    EXPORT_WHEN_ACTIVE = 2
    r"""仅导出处于激活的量化参数"""

    INTERNAL = 3
    r"""不可见，不导出"""


class BuiltinPlatform(IntEnum):
    r"""目标平台。在MPPQ中并不预置任何平台，用户可以通过
    :func:`~mppq.api.register_platform` 注册自定义平台。

    每个自定义平台需要指定：
    1. frontend 模型文件解析器，默认会使用 onnx_parser.
    2. dispatcher 调度器，调度器负责分配每个算子的执行精度。
    3. quantizer 量化器，量化器负责具体的量化算法执行，比如使用哪些
       :class:`~mppq.quantization.optim.base.QuantizationOptimizationPass`;
       量化策略是static还是dynamic，量化位宽是多少，量化精度是多少，等等。
    4. exporter 导出器，默认会使用 onnx_exporter.

    更多的，用户还可以复写某些算子的计算方式，通过 :func:`~mppq.api.register_operation_handler` 可以
    替换默认的算子 forward 函数。
    """

    UNSPECIFIED = -1
    r"""未指定"""


class TargetPrecision(Enum):
    r""":class:`TargetPrecision` 是每个算子被赋予的属性之一。

    它描述了算子的期望执行精度。
    """

    SOI = -1
    r"""Shape Or Index. 实际上该属性代表所有非特征数据的变量。
    比如scale, roi, offset, start, end, axis, etc.
    """

    UNSPECIFIED = 0

    FP32 = 1

    INT8 = 2

    INT8_FP16 = 3
    r"""Mixed INT8 as inputs and FP16 as outputs"""

    INT8_INT4 = 4
    r"""Mixed INT8 as features and INT4 as weights"""


class RoundingPolicy(Enum):
    """RoundingPolicy is a core setting for PPQ quantization calculation. It
    defines rounding behaviour inside quantization calculation.

    Formula: quant(x) = clip(round(x / scale, RoundingPolicy), -128, 127)

    PPQ Supports 7 different rounding policies now.
    Take a look at https://en.wikipedia.org/wiki/Rounding

    ATTENTION: RoundingPolicy greatly affects PPQ executor behaviour in some cases,
        to get a correct result from PPQ executor,
        make sure your RoundingPolicy is the same as your hardware.
    """

    ROUND_HALF_EVEN = 0
    ROUND_HALF_UP = 1
    ROUND_HALF_DOWN = 2
    ROUND_HALF_TOWARDS_ZERO = 3
    ROUND_HALF_FAR_FORM_ZERO = 4
    ROUND_TO_NEAR_INT = 5
    ROUND_UP = 6


class QuantizationProperty(IntEnum):
    r"""量化策略。不同的量化策略可以相互组合成复合策略。

    Note:

        某些冲突策略不能组合。比如 PER_TENSOR 与 PER_CHANNEL 不能组合。
    """

    PER_TENSOR = 1
    PER_CHANNEL = 1 << 1
    LINEAR = 1 << 2
    FLOATING = 1 << 3
    SYMMETRIC = 1 << 4
    ASYMMETRIC = 1 << 5
    POWER_OF_2 = 1 << 6
    DYNAMIC = 1 << 7


class QuantizationPolicy:
    r"""组合量化策略。

    一个量化策略由多个量化属性组成，每个属性可以是 PER_TENSOR 或 PER_CHANNEL。
    一个量化策略可以是线性的，也可以是非线性的。
    一个量化策略可以是静态的，也可以是动态的。
    一个量化策略可以是对称的，也可以是非对称的。
    一个量化策略可以是 2 的幂的，也可以不是 2 的幂的。

    量化策略的组合可以得到更复杂的量化策略。比如，PER_TENSOR + LINEAR + DYNAMIC 就是一个动态线性策略。

    冲突的策略将会抛出异常 AssertionError.
    """

    def __init__(self, policy: int | QuantizationProperty) -> None:
        QuantizationPolicy._check(policy)
        self._policy = int(policy)

    def has_property(self, prop: QuantizationProperty) -> bool:
        r"""判断量化策略是否包含某个属性。"""
        return (self._policy & prop) != 0

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, QuantizationPolicy):
            raise TypeError(
                "Can only compare QuantizationPolicy object "
                "with another QuantizationPolicy object."
            )
        return self._policy == obj._policy

    @staticmethod
    def _check(policy: int):
        # fmt: off
        assert policy in {
            # pylint: disable=line-too-long
            QuantizationProperty.ASYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL, # noqa
            QuantizationProperty.ASYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR, # noqa
            QuantizationProperty.ASYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2, # noqa
            QuantizationProperty.ASYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC, # noqa
            QuantizationProperty.ASYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC, # noqa
            QuantizationProperty.ASYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2, # noqa
            QuantizationProperty.ASYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2, # noqa
            QuantizationProperty.SYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2, # noqa
            QuantizationProperty.ASYMMETRIC | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2, # noqa
        }
        # fmt: on

    def to_dict(self) -> Dict[str, bool]:
        """return a dictionary to describe this policy."""
        return {p.name: self.has_property(p) for p in QuantizationProperty}


class QuantizationStates(Enum):
    r"""参数的量化进度和当前状态."""

    INITIAL = 1
    r"""量化参数刚刚被初始化，当前 config 不生效，数据不能被使用"""

    BAKED = 2
    r"""只针对参数量化，表示参数已经被静态量化，当前 config 不生效，数据可以直接使用"""

    OVERLAPPED = 3
    r"""表示这一路输入不量化，当前量化信息被父量化信息所覆盖"""

    ACTIVATED = 4
    r"""表示当前 config 生效"""

    PASSIVE = 5
    r"""表示这一路输入被动量化，如 bias, clip value 等，被动量化参数使用其他 TQC 的量化信息完成量化"""

    PASSIVE_INIT = 6
    r"""表示这一路输入被动量化，并且刚刚初始化不能被使用"""

    PASSIVE_BAKED = 7
    r"""被动量化且静态量化，当前config不生效，数据可以直接使用"""

    FP32 = 8
    r"""表示这一路输入不量化"""

    @staticmethod
    def is_activated(state: "QuantizationStates") -> bool:
        r"""是否已具有正确的量化参数"""
        return state in {QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE}

    @staticmethod
    def can_export(state: "QuantizationStates") -> bool:
        r"""是否可以导出"""
        return state not in {
            QuantizationStates.INITIAL,
            QuantizationStates.PASSIVE_INIT,
        }


class TensorQuantizationConfig(Serializable):
    r"""Tensor 量化控制结构体
    PPQ 使用量化控制结构体描述量化行为，该结构体被定义在 ppq.core.quant 中。
    截止 PPQ 0.6.6 版本，该结构体由 15 项不同的属性组成。我们将向你介绍这一核心数据结构体的设计构想。

    ## 1. QuantizationPolicy 量化策略

    在 TensorQuantizationConfig 当中，首当其冲地内容是 TQC.policy，这是一个 QuantizationPolicy 对象。
    policy 属性用于描述量化的规则，一个完整的量化策略是由多个量化属性(QuantizationProperty)组合完成的；
    在 PPQ 中目前我们支持 8 种不同的量化属性，你可以使用以下属性来组合形成自定义的量化规则:

        1. PER_TENSOR: 以 Tensor 为单位完成量化，每个 Tensor 使用一个 scale 和 offset 信息。

        2. PER_CHANNEL: 以 Channel 为单位完成量化，每个 Channel 使用一个 scale 和 offset 信息。

        3. LINEAR: 线性量化，通常的 INT8, INT16 皆属于线性量化，在线性量化的表示中不存在指数位。

        4. FLOATING: 浮点量化，包括 FP8 E4M3, FP8 E5M2, FP16, BF16 等格式，在浮点量化中数据由底数和指数两部分组成。

        5. SYMMETRICAL: 对称量化，在量化计算中不启用 offset。

        6. ASYMMETRICAL: 非对称量化，在量化计算中启用 offset 完成量化偏移。

        7. POWER_OF_2: 限制 scale 取值必须为 2 的整数次幂，这一量化行为多见于端侧以及浮点量化。

        8. DYNAMIC: 启用动态量化策略，对于每一批次的数据，scale 与 offset 都将被动态地计算更新。

    下图解释了浮点量化与线性量化的区别：

    ![image](https://user-images.githubusercontent.com/43309460/199235366-1e83ed97-0731-4e1d-abeb-b7121e3d2a94.png)

    ## 2. 线性量化与相关属性

    线性量化允许与下列属性进行组合：参考 :class:`QuantizationPolicy` 。

    线性量化是最为常用的数值量化方法，有些时候我们也称其为均匀量化，在线性量化中，量化操作的计算方法为：

        - Unscaled FP32 = (FP32 / scale) - offset
        - INT8 = Clip(Round(Unscale FP32), quant_min, quant_max)
        - Dequantized FP32 = (INT8 + offset) * scale

    其中 Round 函数行为由 TQC.rounding(RoundingPolicy) 属性确定，PPQ 支持 7 种不同的取整策略，
    其中 ROUND_HALF_EVEN 是最常见的取整策略，关于取整策略的详细讨论可以参考
    https://en.wikipedia.org/wiki/Rounding

    quant_min, quant_max 分别由 TQC.quant_min, TQC.quant_max 属性确定，对于线性量化而言他们是整数，
    通常为[-128, 127]。部分框架使用 [-127, 127] 作为截断值，在部分场景下如此定义将有优势，
    但在 Onnx 的 Q/DQ 算子定义中不允许使用 [-127, 127] 作为截断。

    PPQ 可以模拟 1-32 bit 的任意位宽量化，但若以部署为目的，不建议使用 8 bit 之外的配置。
    用户须知高位宽量化可能造成 Scale 过小，以至于浮点下溢出。

    ## 3. 浮点量化与相关属性

    浮点量化允许与下列属性进行组合：参考 :class:`QuantizationPolicy` 。

    在浮点量化中，量化函数的计算方法为：

        - Unscaled FP32 = (FP32 / scale)
        - FP8 = Convert(Unscale FP32, quant_min, quant_max)
        - Dequantized FP32 = FP8 * scale

    其中 Convert 函数行为复杂，其转换过程分为三种不同的情况：

        - 当 Unscaled FP32 大于 quant_max，或者小于 quant_min，则直接进行截断
        - 当 Unscaled FP32 幅值大于 FP8 能够表达的最小值，此时需要移去多余的底数位，并对底数进行四舍五入
        - 当 Unscaled FP32 数据小于规范化 FP8 能够表达的最小值，此时浮点下溢出，此时我们计算
          `FP8 = Round(Unscaled FP32 / FP8_min) * FP8_min`

    其中 FP8_min 是非规格化 FP8 能够表达的最小值。对于 FP8 E4M3 标准而言，其能表示的最大值为 448.0，最小值为 -448.0。

    quant_min, quant_max 分别由 TQC.quant_min, TQC.quant_max 属性确定，对于 FLOATING 量化，
    我们引入一个新的属性 TQC.exponent_bits(int)。使用这个属性来指定总位宽中有多少数位用于表示指数
    (相应地，底数位为总位宽-指数位-1)。

    在浮点量化中，尺度因子的选取对量化效果的影响不大，因此用户可以使用 constant 校准策略
    (见 ppq.quantization.observer)将所有尺度因子设置为1。

    关于浮点量化的具体细节可以参考 [本文](https://zhuanlan.zhihu.com/p/574825662)

    ## 4. 其他量化控制属性

    1. TQC.num_of_bits(int)：量化位宽，对于 INT8, FP8 量化，量化位宽为 8。对于 INT16, FP16 量化，量化位宽为16。
    2. TQC.state(QuantizationStates): 量化状态，在 PPQ 中目前有共计 8 种不同的量化状态，该属性极大地丰富了 PPQ
       量化信息的语义，使得我们能够更加灵活地控制量化行为。该属性可以被用于切换 量化 / 非量化 状态；执行量化联合定点；执行参数烘焙。
    3. TQC.channel_axis(int): 量化轴，对于 PER_CHANNEL 量化，使用这个属性来指定沿着那一维度展开量化，如执行 Per-tensor
       量化，该属性被忽略，用户可以将其设置为 None。
    4. TQC.observer_algorithm(str): observer 算法，其中 observer 是用于确定 scale 和 offset 的对象，
       使用这个属性指明要使用何种类型的 observer 确定 scale 和 offset
    5. TQC.dominator(TensorQuantizationConfig): 一个指向父量化信息的指针。在 PPQ 中 TQC 与 TQC 之间并不是独立的，
       他们之间可以存在父子关系。所有子量化信息与父量化信息共享 scale 和 offset
    6. TQC.visibility(QuantVisibility): 导出可见性，使用这个属性来告知 ppq 的导出器是否需要导出当前的 TQC。

    ## 5. 量化控制结构体的初始化

    TensorQuantizationConfig 是 PPQ 中的核心数据结构，它总是由 Quantizer 对象完成创建的：

    .. code-block:: python

        # 下面这段代码为一个指定的算子创建了相应的 Tensor Quantization Config
        quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_FP8, graph=graph)
        # 取得 TRT_FP8 所对应的量化器
        quantizer.quantize_operation(op_name = op.name, platform = dispatching[op.name])

    在 PPQ 当中，Quantizer 的职责即是为算子初始化他们的量化控制结构体。
    不同的量化器将按照不同的规则创建控制结构体，如 TRT_FP8 所对应的量化器 只会为了 Conv, Gemm 算子创建量化信息，
    要求他们的输入按照对称-浮点-Per Channel的方式完成量化。
    而 DSP_INT8 所对应的量化器为几乎所有算子创建量化信息，要求他们按照非对称-线性-Per Tensor的方式完成量化。

    用户可以手动创建量化控制结构体，使用 ppq.lib 中的接口：

        # 创建一个默认的线性量化控制结构体(对称, per-tensor)
        from ppq.lib import LinearQuantizationConfig
        TQC = LinearQuantizationConfig()

        # 创建一个默认的浮点量化控制结构体(FP8 E4M3)
        from ppq.lib import FloatingQuantizationConfig
        TQC = FloatingQuantizationConfig()

    ## 6. 量化控制结构体的校准

    绝大部分的 TensorQuantizationConfig 在完成初始化之后都无法使用-他们的 scale 与 offset 均为空值，
    且 Quantizer 在初始化他们时会将其状态(TQC.state)置为 INITIAL，处于这个状态的量化信息在计算过程中不会被启用。

    我们必须送入一定量数据，进行必要 Calibration 操作后才能为网络中的量化信息确定合理的 scale 与 offset 值，
    这一过程是由种类繁多的 Observer 完成的：

    .. code-block:: python

        # PPQ 目前支持 8 种不同的 Observer
        OBSERVER_TABLE = {
            'minmax': TorchMinMaxObserver,
            'kl': TorchHistObserver,
            'percentile': TorchPercentileObserver,
            'mse': TorchMSEObserver,
            'isotone': TorchIsotoneObserver,
            'constant': ConstantObserver,
            'floating': DirectMSEObserver,
            'isotone': ...
        }

    这些 Observer 会负责在网络计算过程中收集必要的统计信息，并为 TQC 的 scale 与 offset 赋予有效的值。
    在完成一切之后，Observer 还会负责将 TQC 的状态(TQC.state)修改为 ACTIVATED。此时量化信息将被正式启用，
    从而在网络前向传播模拟量化计算。

    关于 Observer 的讨论，可以参考 [本视频](https://www.bilibili.com/video/BV1QF41157aM)

    ## 7. 量化控制结构体的父子链接

    在我们讨论量化时，对于那些存在着多个输入的算子，例如 add, concat，它们的所有输入总是被要求有着相同的 scale。
    为了表述这种语义，我们为 TQC 添加了 TQC.dominator 属性，这一属性可以指向另一个量化控制结构体。

    假设我们存在两个不同的量化控制结构体 A, B：

    - 语句 A.dominator = B 表示 A 将共享 B 的 scale 与 offset(A.policy, A.num_of_bits等属性仍将使用自己的)。
      于此同时 A.state 将被修改为 OVERLAPPED(A 将不再启用)
    - 语句 A.master = B 表示 A 将共享 B 的 scale 与 offset(A.policy, A.num_of_bits等属性仍将使用自己的)。
      于此同时 A.state 将被修改为 PASSIVE(A 将仍然启用，但不具有独立的量化参数)

    如果 A 已经是其他量化结构体 C 的父节点，则上述过程将级联地使得 B 成为 A, C 共同的父节点，A, C 都将共享 B 的 scale 与 offset。

    下图简述了在量化控制结构体的生命周期中，量化状态是如何变迁的
    （[量化优化过程](https://github.com/openppl-public/ppq/tree/master/ppq/quantization/optim)将负责修改量化控制信息的状态）：

    ![Quantization State](https://user-images.githubusercontent.com/43309460/199236632-ec69ca29-9900-4875-8299-a196546d0dde.png)
    """

    def __init__(
        self,
        policy: QuantizationPolicy,
        rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN,
        num_of_bits: int = 8,
        quant_min: int = -127,
        quant_max: int = 128,
        exponent_bits: int = 0,
        scale: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        observer_algorithm: str = "minmax",
        detail: Optional[Any] = None,
        channel_axis: int = 0,
        visibility: QuantVisibility = QuantVisibility.EXPORT_WHEN_ACTIVE,
        state: QuantizationStates = QuantizationStates.INITIAL,
    ):
        r"""Create a PPQ Tensor Quantization Configuration Instance.

        Args:
            policy (QuantizationPolicy): Quantization policy instance which defines the
                quantization behaviour from marco view.
            rounding (RoundingPolicy): Rounding policy used in quantization.
            num_of_bits (int): Quantization fraction bits. (2 < num_of_bits < 32)
            exponent_bits (int): Quantization exponent bits. (0 < num_of_bits < 8)
                For Int8 Quantization, num_of_bits = 8 and exponent_bits = 0
                For FP8 Quantization, num_of_bits = 4 and exponent_bits = 4
            quant_min (int): An integer value represents the upper bound(inclusive) of
                quantized value.
            quant_max (int): An integer value represents the lower bound(inclusive) of
                quantized value.
            scale (Any): Scale of quantized value, for per-tensor quantization policy,
                we use a single float as its scale, while for per-channel quantization
                policy, it will be an array that contains scales for each channel.
            offset (Any): Quantization offset for ASYMMETRICAL quantization policy,
                it will be set as 0 in SYMMETRICAL quantization schema.
            observer_algorithm (str): A string represents an observing algorithm for
                this tensor.
            detail (Any, optional): Only used by PPQ internal logic, detail is used to
                store some internal data, you are not supposed to use it.
            channel_axis (int, optional): Only used in PER_CHANNEL quantization,
                channel index.
            visibility (Visibility): visibility is the attribute that controls export
                logic. Currently, there are 3 Visibility level in PPQ:
                if Visibility == FORCE_EXPORT, ppq exporter will export this TQC
                    ignoring state check(even if current TQC has been overrlapped).
                if Visibility == EXPORT_WHEN_ACTIVD, ppq exporter will export this TQC
                    only when it has been activated.
                if Visibility == INTERNAL, This TQC will not be exported.
            state (QuantizationStates, optional): Defaults to
                QuantizationStates.INITIAL, see QuantizationStates for more detail.
        """

        assert num_of_bits <= 32, "Cannot quantize a tensor with more than 32 bits."
        assert num_of_bits >= 2, "Cannot quantize a tensor with less than 2 bits."
        assert (
            exponent_bits <= 8
        ), "Cannot quantize a tensor with more than 8 bits exponent(fp32 overflow)."
        assert (
            exponent_bits >= 0
        ), "Cannot quantize a tensor with less than 0 bits exponent."

        self._exponent_bits = exponent_bits
        self.policy = policy
        self.num_of_bits = num_of_bits
        self.state = state
        self.observer_algorithm = observer_algorithm
        self._scale = scale
        self._offset = offset
        self._rounding = rounding
        self._quant_min = quant_min
        self._quant_max = quant_max
        self._channel_axis = channel_axis
        self._dominator = self  # union-find
        self._hash = self.__create_hash()
        self.visibility = visibility
        self.detail = {} if detail is None else detail
        super().__init__()

    def can_export(self, export_overlapped: bool = EXPORT_OVERLAPPED_CONFIG) -> bool:
        if self.visibility == QuantVisibility.INTERNAL:
            return False
        valid_states = {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}

        if export_overlapped:
            valid_states.add(QuantizationStates.OVERLAPPED)
        state_check = (
            QuantizationStates.is_activated(self.state) or self.state in valid_states
        )

        if state_check or self.visibility == QuantVisibility.FORCE_EXPORT:
            if self.is_observed():
                return True
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TensorQuantizationConfig):
            raise TypeError(
                "Can only compare TensorQuantizationConfig object "
                "with another TensorQuantizationConfig object."
            )
        return self._hash == o._hash

    def __str__(self) -> str:
        return f"PPQ TensorQuantizationConfig({self.__hash__()})"

    _hash_seed = int(time.time())

    @staticmethod
    def __create_hash():
        TensorQuantizationConfig._hash_seed = (
            0x343FD * TensorQuantizationConfig._hash_seed + 0x269EC3
        ) % (2 << 31)
        return TensorQuantizationConfig._hash_seed

    def __hash__(self) -> int:
        return self._hash

    def is_same_scheme(self, o: "TensorQuantizationConfig") -> bool:
        if not isinstance(o, TensorQuantizationConfig):
            raise TypeError(
                "Can only compare TensorQuantizationConfig object "
                "with another TensorQuantizationConfig object."
            )
        return (
            self.quant_max == o.quant_max
            and self.quant_min == o.quant_min
            and self.policy == o.policy
            and self.num_of_bits == o.num_of_bits
            and self.exponent_bits == o.exponent_bits
            and self.channel_axis == o.channel_axis
            and self.rounding == o.rounding
        )

    @property
    def dominated_by(self) -> "TensorQuantizationConfig":
        r"""dominated_by is a crucial feature for tensor quantization
        configuration in PPQ. This property is actually maintained by union-
        find set data structure.

        Every tensor quantization configuration(A) is created with dominated_by = self,
        and only when it is overlapped by other configuration(B), it shall set
        A.dominated_by = B. Setting A.dominated_by = B also makes A, B as a quantization
        group. (quantization state of A is always set as OVERLAPPED here)

        So to say every tensor quantization configuration with dominated_by != self is
        overrlaped by other quantization configuration. When a tensor quantization
        configuration is overlapped, it means this tensor is already been quantized
        with another quantization configuration, and there is no need to be quantized
        with this configuration anymore.

        PPQ use this property to find root configuration for each configuration group,

        Returns:
            [TensorQuantizationConfig]: root configuration of this quantization group.

        ATTENTION: This configuration is invalid when self.dominated_by != self.
        """
        if self._dominator == self:
            return self
        else:
            root = self._dominator.dominated_by
            self._dominator = root
            return root

    @dominated_by.setter
    def dominated_by(self, obj: "TensorQuantizationConfig"):
        assert isinstance(obj, TensorQuantizationConfig)
        if obj == self:
            raise ValueError(
                "Error with TQC.dominated_by = obj: obj mustn't equal to its self."
            )
        root, dominator = self.dominated_by, obj.dominated_by
        if dominator == self:
            raise ValueError(
                "Can not Assign Dominator like this, "
                "Circular reference was detected. Son TQC can not dominate its Father."
            )
        assert isinstance(root, TensorQuantizationConfig)
        if dominator != root:
            root._dominator = dominator  # pylint: disable=protected-access
            self._dominator = dominator
            root.state = QuantizationStates.OVERLAPPED
            self.state = QuantizationStates.OVERLAPPED

    @property
    def master_by(self) -> "TensorQuantizationConfig":
        if self._dominator == self:
            return self
        else:
            root = self._dominator.dominated_by
            self._dominator = root
            return root

    @master_by.setter
    def master_by(self, master: "TensorQuantizationConfig"):
        assert isinstance(master, TensorQuantizationConfig)
        if master == self:
            raise ValueError(
                "Error with TQC.dominated_by = o: o must not equal to TQC its self."
            )
        self._dominator = master
        if master.scale is not None and master.offset is not None:
            self.state = QuantizationStates.PASSIVE
        else:
            self.state = QuantizationStates.PASSIVE_INIT

    def is_revisable(self):
        return self.dominated_by == self and self.state in {
            QuantizationStates.ACTIVATED,
            QuantizationStates.FP32,
            QuantizationStates.FP32,
            QuantizationStates.INITIAL,
            QuantizationStates.FP32,
            QuantizationStates.PASSIVE,
            QuantizationStates.PASSIVE_INIT,
        }

    def is_observed(self) -> bool:
        """Whether scale is initialized or not."""
        try:
            return self.scale.numel() > 0
        except ValueError:
            return False

    @property
    def scale(self) -> torch.Tensor:
        """Get Quantization Scale of this TQC.

        If current TQC is dominated by other, return father TQC's scale instead.
        """
        if self.dominated_by == self:
            if self._scale is None:
                raise ValueError("scale is not initialized.")
            return self._scale
        else:
            return self.dominated_by.scale

    @property
    def offset(self) -> torch.Tensor:
        """Get Quantization Offset of this TQC.

        If current TQC is dominated by other, return father TQC's offset instead.
        """
        if self.dominated_by == self:
            if self._offset is None:
                raise ValueError("offset is not initialized.")
            return self._offset
        else:
            return self.dominated_by.offset

    @property
    def rounding(self) -> RoundingPolicy:
        """Get Rounding Policy of this TQC."""
        return self._rounding

    @property
    def quant_min(self) -> int:
        """Get minimum quant value of this TQC."""
        return self._quant_min

    @property
    def quant_max(self) -> int:
        """Get maximum quant value of this TQC."""
        return self._quant_max

    @property
    def exponent_bits(self) -> int:
        """Get exponent bit-width of current TQC.

        num_of_bits = exponent_bits + mantissa_bits
        """
        return self._exponent_bits

    @property
    def mantissa_bits(self) -> int:
        """Get mantissa bit-width of current TQC.

        num_of_bits = exponent_bits + mantissa_bits
        """
        # there is one bit for sign.
        return self.num_of_bits - self._exponent_bits - 1

    @property
    def channel_axis(self) -> int:
        """Get Quantization Axis, For Per-tensor Quantization, it returns None."""
        return self._channel_axis

    @channel_axis.setter
    def channel_axis(self, channel_axis: int):
        if not self.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError(
                "Can not change property: quantization channel axis for this TQC. "
                "self.policy.has_property(QuantizationProperty.PER_CHANNEL) == False."
            )
        self._channel_axis = channel_axis

    @scale.setter
    def scale(self, value: Optional[torch.Tensor]):
        if not self.is_revisable():
            raise PermissionError(
                "Can not change scale of this tensor quantization configuration now. "
                "It has been overlapped or has an inactive state. "
                "Due to it is not a active config, any change of this configuration "
                "is not allowed."
            )
        else:
            self._scale = value

    @offset.setter
    def offset(self, value: Optional[torch.Tensor]):
        if not self.is_revisable():
            raise PermissionError(
                "Can not change offset of this tensor quantization configuration now. "
                "It has been overlapped or has an inactive state. "
                "Due to it is not a active config, any change of this configuration is "
                "not allowed."
            )
        else:
            self._offset = value

    @rounding.setter
    def rounding(self, policy: RoundingPolicy):
        self._rounding = policy

    @quant_min.setter
    def quant_min(self, value: int):
        self._quant_min = value

    @quant_max.setter
    def quant_max(self, value: int):
        self._quant_max = value

    @exponent_bits.setter
    def exponent_bits(self, bits: int):
        if not self.policy.has_property(QuantizationProperty.FLOATING):
            raise PermissionError(
                "Can not change property: exponent bits for this TQC. "
                "self.policy.has_property(QuantizationProperty.FLOATING) == False."
            )
        self._exponent_bits = bits

    def __deepcopy__(self, memo=None) -> "TensorQuantizationConfig":
        """Create a tensor config from this one, keep policy and state
        unchanged.

        if there is an non-empty scale and offset, they will be cloned too.
        """
        scale, offset = None, None
        if self.scale is not None:
            scale = deepcopy(self.scale, memo)
        if self.offset is not None:
            offset = deepcopy(self.offset, memo)
        config = TensorQuantizationConfig(
            policy=self.policy,
            rounding=self.rounding,
            num_of_bits=self.num_of_bits,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            scale=scale,
            offset=offset,
            observer_algorithm=self.observer_algorithm,
            detail=deepcopy(self.detail, memo),
            state=self.state,
            exponent_bits=self.exponent_bits,
            channel_axis=self.channel_axis,
            visibility=self.visibility,
        )
        if self.state == QuantizationStates.OVERLAPPED:
            config._dominator = self._dominator
        return config


class OperationQuantizationConfig:
    """OperationQuantizationConfig serves as a collection of tensor
    quantization configuration.

    See TensorQuantizationConfig for more information.
    """

    def __init__(
        self,
        input_quantization_configs: Optional[List[TensorQuantizationConfig]] = None,
        output_quantization_configs: Optional[List[TensorQuantizationConfig]] = None,
        is_positive_quant_op: bool = True,
    ):
        """Create an operation quantization configuration.

        Args:
            input_quantization_configs (List[TensorQuantizationConfig], optional):
                a list contains all configuration of all input variables.

            output_quantization_configs (List[TensorQuantizationConfig], optional):
                a list contains all configuration of all output variables.

            ATTENTION: whether a variable is gonna to be quantized or not, it must have
            a quantization configuration.

            is_positive_quant_op (bool, optional): [description]. Defaults to True.
                some operations are passively quantized, such as Maxpooling, Padding.
                For those operations, set this property as False, PPQ will use this
                property to optimize your graph.
        """
        self.input_quantization_config = self._check_famliy_config(
            input_quantization_configs or []
        )
        self.output_quantization_config = self._check_famliy_config(
            output_quantization_configs or []
        )
        self.is_active_quant_op = is_positive_quant_op

    def _check_famliy_config(self, famliy_configs: List[TensorQuantizationConfig]):
        for famliy_config in famliy_configs:
            if not isinstance(famliy_config, TensorQuantizationConfig):
                raise TypeError(
                    f"You are trying to set famliy quantization config of {str(self)}, "
                    "except one TensorQuantizationConfig object, "
                    f"however got a {type(famliy_config)}."
                )
        return famliy_configs

    def __iter__(self) -> Iterator[TensorQuantizationConfig]:
        for cfg in self.input_quantization_config + self.output_quantization_config:
            yield cfg

    def __len__(self):
        size = len(self.input_quantization_config)
        size += len(self.output_quantization_config)
        return size

    def __repr__(self) -> str:
        return (
            f"Inputs config: {self.input_quantization_config}, "
            f"Outputs config {self.output_quantization_config}"
        )

    def __deepcopy__(self, memo=None):
        """Create an operation config from this one, keep policy and state
        unchanged.

        if this one has an non-empty scale or offset, they will be cloned too.
        """
        return OperationQuantizationConfig(
            input_quantization_configs=[
                deepcopy(i, memo) for i in self.input_quantization_config
            ],
            output_quantization_configs=[
                deepcopy(i, memo) for i in self.output_quantization_config
            ],
            is_positive_quant_op=self.is_active_quant_op,
        )
