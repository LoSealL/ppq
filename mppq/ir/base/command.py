from enum import Enum

from mppq.ir.base.opdef import Operation, Variable
from mppq.quant import OperationQuantizationConfig, TargetPrecision


class GraphCommandType(Enum):
    r"""预置的图变换操作"""

    # convert value inside graph to torch.tensor(usually from numpy)
    CONVERT_TO_TENSOR = 0

    # 图上权重部署到 GPU(tensor)，由 RunnableGraph 进行处理
    # deploy graph weights to GPU
    DEPLOY_TO_CUDA = 1

    # 图上权重部署到 CPU(tensor)，由 RunnableGraph 进行处理
    # deploy graph weights to CPU, in tensor format
    DEPLOY_TO_CPU = 2

    # 图上权重部署到 CPU(tensor)，由 RunnableGraph 进行处理
    # deploy graph weights to CPU, in array format
    DEPLOY_TO_NUMPY = 3

    # 量化一个指定OP，同时将所有关联的 variable 转换为量化 variable
    # quantize a specified operation, and converts all connected
    # variables to quantable variables
    QUANTIZE_OPERATION = 5

    # 将一个OP的量化暂时解除，同时将所有关联的 variable 解除量化
    # deactivate quantization state of an op temporarily,
    # and deactivated all related variables
    DISABLE_OPERATION_QUANTIZATION = 6
    # 将一个OP的量化状态恢复
    # restore quantization state of a dequantized op
    RESTORE_OPERATION_QUANTIZATION = 7

    # 格式化 CLIP 算子，将不同行为的 CLIP 行为统一
    # regularize Clip operator
    FORMAT_CLIP = 9
    # 格式化 PAD 算子，将不同行为的 PAD 行为统一
    # regularize Pad operator
    FORMAT_PAD = 10
    # 格式化 GATHER 算子，将 index 参数（由input输入）植入算子属性
    # regularize gather operator
    FORMAT_GATHER = 11
    # 格式化 CAST 算子，统一 CAST 参数到 PPQ.core.DataType
    # regularize Cast operator
    FORMAT_CAST = 12
    # 格式化所有常量输入，尝试将他们转换为int32的
    # regularize all constant inputs
    FORMAT_INT64_CONSTANT = 13
    # 移除所有孤立节点
    # remove all isolated operators
    DELETE_ISOLATED = 14
    # 将所有参数变量进行分裂（只允许一个 dest_op ）
    # split variables and each variable is allowed to be used by only one operator
    FORMAT_PARAMETERS = 16

    # 用一个新的算子替换一个原来的
    # replace an old op with a new one
    REPLACE_OP = 17
    # 用一个新的 var 替换一个原来的
    # replace an old variable with a new one
    REPLACE_VAR = 18
    # 移除一个算子的输入参数，只能移除 Parameter
    # remove input parameters of am operator
    REMOVE_INPUT = 19

    # 图遍历模式匹配（要求返回路径）
    # graph traversal pattern matching(return paths)
    TRAVERSAL_PATTERN_MATCHING = 20
    # 图遍历模式匹配（仅要求返回点集）
    # graph traversal pattern matching(return op set)
    TRAVERSAL_OPSET_MATCHING = 21
    # 激活函数匹配
    # activation op matching
    ACTIVATION_MATCHING = 22
    # Concat 匹配
    # Concat op matching
    CONCAT_MATCHING = 23

    # 插入 Device Switcher
    # insert switcher
    INSERT_SWITCHER = 25
    # 移除 Device Switcher
    # remove switcher
    REMOVE_SWITCHER = 26

    # 融合图中的 计算层 与 BN
    # fuse Computing layer and BN
    FUSE_BN = 27
    # 删除图中的 Constant Input
    # remove constant input
    FORMAT_CONSTANT_INPUT = 28
    # 将 opset1 的 slice 弄成 opset 11 的
    FORMAT_SLICE = 29
    # 从一个指定位置将图截断
    TRUNCATE_ON_VAR = 30

    # 升级图中的 resize 到 opset 11
    FORMAT_RESIZE = 31

    # Replace Single Batchnorm to some op else.
    REPLACE_BATCHNORM_TO_CONV = 32
    REPLACE_BATCHNORM_TO_SCALE = 33

    # Fuse Bias add to Gemm, Conv and ConvTranspose
    FUSE_BIAS_ADD = 34

    # 移除 Identity
    # remove all identity ops from your graph
    REMOVE_IDENTITY = 35


class GraphCommand:
    """A graph transformation command with type and arguments."""

    def __init__(self, command_type: GraphCommandType, **kwargs) -> None:
        assert isinstance(command_type, GraphCommandType)
        self.command_type = command_type
        self.kwargs = kwargs

    def __str__(self) -> str:
        return (
            f"GraphCommand object {self.__hash__()},\t "
            f"Command type: {self.command_type},\t Args:{self.kwargs}"
        )


class GraphDeployCommand(GraphCommand):
    """Command to deploy graph to a device."""

    def __init__(self, device: str) -> None:
        if device.startswith("cuda"):
            super().__init__(GraphCommandType.DEPLOY_TO_CUDA)
        elif device.startswith("cpu"):
            super().__init__(GraphCommandType.DEPLOY_TO_CPU)
        else:
            raise ValueError(f"Device type {device} not understand.")
        self.device = device


class QuantizeOperationCommand(GraphCommand):
    r"""将普通算子切换为量化封装算子."""

    def __init__(
        self,
        op_name: str,
        target_precision: TargetPrecision,
        config: OperationQuantizationConfig,
    ) -> None:
        super().__init__(command_type=GraphCommandType.QUANTIZE_OPERATION)
        self.op_name = op_name
        self.target_precision = target_precision
        self.config = config


class ReplaceOperationCommand(GraphCommand):
    r"""置换算子."""

    def __init__(self, op_name: str, replace_to: Operation) -> None:
        super().__init__(command_type=GraphCommandType.REPLACE_OP)
        self.op_name = op_name
        self.replace_to = replace_to


class ReplaceVariableCommand(GraphCommand):
    r"""置换变量."""

    def __init__(self, var_name: str, replace_to: Variable) -> None:
        super().__init__(command_type=GraphCommandType.REPLACE_VAR)
        self.op_name = var_name
        self.replace_to = replace_to


class TruncateGraphCommand(GraphCommand):
    r"""将计算图截断至指定的变量处."""

    def __init__(self, var: Variable, mark_as_output: bool) -> None:
        super().__init__(command_type=GraphCommandType.TRUNCATE_ON_VAR)
        self.var = var
        self.mark_as_output = mark_as_output
