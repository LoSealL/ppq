# PPQ 全局配置，你可以自由修改下列属性以达成特定目的。
# PPQ System configuration
# You can modify following codes for your own purpose.

# Observer 中，最小 scale 限制，所有小于该值的 scale 将被该值覆盖
OBSERVER_MIN_SCALE = 1e-4
# Observer 中，最小 scale 的手动覆盖属性
OBSERVER_MIN_SCALE_MANUL_OVERRIDE = "OBSERVER_MIN_SCALE_MANUL_OVERRIDE"
# Observer 中 kl 散度的计算设备
OBSERVER_KL_COMPUTING_DEVICE = "cpu"
# Observer 中 hist 箱子的个数
OBSERVER_KL_HIST_BINS = 4096
# Observer 中 hist 箱子的属性
OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE = "OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE"
# Observer 中 percentile 的参数
OBSERVER_PERCENTILE = 0.9999
# Observer 中 percentile 参数的属性
OBSERVER_PERCENTILE_MANUL_OVERRIDE = "OBSERVER_PERCENTILE_MANUL_OVERRIDE"
# Observer 中 mse 校准方法 hist 箱子的个数
OBSERVER_MSE_HIST_BINS = 2048
# Observer 中 mse 计算的间隔，间隔越小，所需时间越长
OBSERVER_MSE_COMPUTE_INTERVAL = 8
# Floating MSE Observer 的样本个数
OBSERVER_FLOATING_MSE_FETCHES = 4096
# Isotone Observer 的监听轴
# 对于 Softmax 激活函数而言，Isotone Observer 的监听轴应该与 Softmax 操作所规约的轴相同
# 对于 Sigmoid 激活函数而言，Isotone Observer 的监听轴应该设置为 Batch 所在的轴
OBSERVER_ISOTONE_OBSERVER_AXIS = "OBSERVER_ISOTONE_OBSERVER_AXIS"

# PASSIVE OPERATIONS 是那些不参与计算的 Op, 这些 op 的输入与输出将直接共享 scale
# 同时这些 op 前后的定点过程将被直接停用
PASSIVE_OPERATIONS = {
    "MaxPool",
    "GlobalMaxPool",
    "Reshape",
    "Flatten",
    "Identity",
    "Dropout",
    "Slice",
    "Pad",
    "Split",
    "Transpose",
    "Interp",
    "Squeeze",
    "Unsqueeze",
    "Resize",
}
# COPUTING OP 是所有带参数的计算层，该属性被用于联合定点和子图切分
COMPUTING_OP = {
    "Conv",
    "Gemm",
    "ConvTranspose",
    "MatMul",
    "Attention",
}
# Activation 是所有激活层，该属性被用于联合定点和子图切分
ACTIVATION_OP = {
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "Tanh",
    "Clip",
    "PRelu",
    "Elu",
    "HardSigmoid",
    "HardSwish",
    "SiLU",
    "Mish",
    "Swish",
    "Gelu",
}
# 默认量化的算子类型
DEFAULT_QUANTIZE_OP = PASSIVE_OPERATIONS | COMPUTING_OP | ACTIVATION_OP

# 强制联合定点的算子种类
TYPES_FOR_ALIGNMENT = {
    "Elementwise": {"Add", "Sub", "Sum"},
    "Concat": {"Concat"},
    "Pooling": {"AveragePool", "GlobalAveragePool"},
}
# 强制联合定点手动覆盖
ALIGNMENT_MANUL_OVERRIDE = "ALIGNMENT_MANUL_OVERRIDE"

# ONNX 导出图的时候，opset的版本，这玩意改了可能就要起飞了
ONNX_EXPORT_OPSET = 19
# ONNX 导出图的时候，onnx version，这玩意改了可能就要起飞了
ONNX_VERSION = 9
ONNX_DOMAIN = "ai.onnx"

DEFAULT_OPSET_DOMAIN = ONNX_DOMAIN
DEFAULT_OPSET_VERSION = ONNX_EXPORT_OPSET
STRICT_OPSET_CHECKING = False

# LSTM 算子的权重缓存属性
LSTM_FLATTEN_WEIGHT_ATTRIB = "LSTM_FLATTEN_WEIGHT_ATTRIB"
# GRU 算子的权重缓存属性
GRU_FLATTEN_WEIGHT_ATTRIB = "GRU_FLATTEN_WEIGHT_ATTRIB"
# 图上用于表示 Opset 的属性
GRAPH_OPSET_ATTRIB = "GRAPH_OPSET"

# LINEAR ACTIVATIONS 是所有线性激活层，PPQ 将执行计算层与线性激活层的联合定点，不论后端是否真的做了图融合。
# 事实上就算后端不融合这些层，执行联合定点也是有益无害的。
LINEAR_ACTIVATIONS = {"Relu", "Clip"}

# 误差容忍度
CHECKPOINT_TOLERANCE = 1

# 要做 Bias Correction 的算子种类
BIAS_CORRECTION_INTERST_TYPE = {"Conv", "Gemm", "ConvTranspose"}

# 导出 qdq 节点时是否需要导出状态已经是 overlap 的节点
EXPORT_OVERLAPPED_CONFIG = False

# 可以产生SOI数据的算子
SOI_DATA_GENERATOR_OPS = {
    "Constant",
    "ConstantOfShape",
    "NonMaxSuppression",
    "Shape",
    "TopK",
}

# 支持整型计算的算子。可以用来传播SOI。
INT_CALCULATION_OPS = {
    "Reshape",
    "Slice",
    "Gather",
    "Pad",
    "Resize",
    "Split",
    "TopK",
    "Tile",
    "Expand",
    "RoiAlign",
    "MMCVRoiAlign",
}
