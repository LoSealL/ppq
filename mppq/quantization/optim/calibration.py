from itertools import cycle
from typing import Any, Callable, Dict, Iterable, List, Optional

from tqdm import tqdm

from mppq.common import OBSERVER_ISOTONE_OBSERVER_AXIS
from mppq.defs import empty_ppq_cache
from mppq.executor import BaseGraphExecutor, RuntimeHook
from mppq.ir.base.graph import BaseGraph
from mppq.ir.base.quantize import QuantableOperation, QuantableVariable
from mppq.logger import debug, error
from mppq.quant import QuantizationStates
from mppq.quantization.observer.base import OBSERVER_TABLE, OperationObserver
from mppq.quantization.optim.base import OPTIM_ALGORITHMS, QuantizationOptimizationPass


@OPTIM_ALGORITHMS.register()
class RuntimeCalibrationPass(QuantizationOptimizationPass):
    r"""Runtime Calibration Pass(量化参数校准过程)

    For integer quantization, you need to calibrate or estimate the scale of all
    floating-point tensors in the model.

    Formula:

            Quant(Y, scale_Y) = Clip(Round(Y / scale_Y))

            Dequant(Y, scale_Y) = Y * scale_Y

    Only activations that have quantization state = INITIAL are going to be calibrated
    via this optimization pass. While if the parameter "override" is set to True,
    activations with quantization state = ACTIVATED will also be re-calibrated.

    Runtime Calibration Pass will write estimated scales and offsets to tensor
    quantization configs, and set their state to ACTIVATED.

    Unlike constant tensors such as weights and biases, variable tensors such as model
    input, activations (outputs of intermediate layers) and model output cannot be
    calibrated unless we run a few inference cycles.

    As a result, PPQ Runtime Calibration Pass requires a representative dataset to
    calibrate them.

    This dataset is supposed to be a small subset (around ~100-500 samples) of the
    training or validation data.

    ### Parameters:

    * method(str):

            String that representing the algorithm used to estimate scales and offsets
            for activations.

            Can be mse, kl, percentile, minmax, this parameter is case insensitive.

            You can register your own calibration method through functions in ppq.api

    * override(bool)

            if this parameter is set to True, activations with quantization
            state = ACTIVATED will also be re-calibrated,
            runtime calibration pass will overwrite their scales and offsets.

            This parameter is introduced since ppq 0.6.4

    ### observer support matrix:

    | observer   | Symmetric | Asymmetric | Per-chanel | Per-tensor | Cuda |
    | ---------- | --------- | ---------- | ---------- | ---------- | ---- |
    | minmax     | [x]       | [x]        | [x]        | [x]        | [ ]  |
    | mse        | [x]       | [x]        | [ ]        | [x]        | [x]  |
    | percentile | [x]       | [x]        | [x]        | [x]        | [x]  |
    | kl         | [x]       | [ ]        | [ ]        | [x]        | [x]  |
    | isotone    | [x]       | [x]        | [ ]        | [x]        | [ ]  |

    ### Usage:

    Runtime Calibration Pass should be invoked before Passive Parameter Quantize Pass

    This pass is included in PPQ Quantization Setting, you can calling this
    optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.quantize_activation = True

        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE,
        collate_fn=collate_fn)

    You can manually create this optimization by:

        from mppq import RuntimeCalibrationPass

        optim = RuntimeCalibrationPass()

    ### Register Calibration Method:

    Using api function register_calibration_observer to resister new observer algorithm
    to PPQ system. Once Algorithm is registered, Runtime Calibration Pass will
    automatically calling them by name.

    This feature requires PPQ > 0.6.5

    """

    def __init__(
        self,
        method: Optional[str] = None,
        override: bool = False,
        calib_steps: int = 32,
    ) -> None:
        super().__init__()
        self._method = method
        self._observers = {}
        self._collate_fn: Optional[Callable[[Any], Any]] = None
        self._steps = calib_steps
        self._override = override

    def calibrate(
        self,
        desc: str,
        dataloader: Iterable[Any],
        executor: BaseGraphExecutor,
        hooks: Dict[str, RuntimeHook],
        output_names: Optional[List[str]] = None,
    ):
        gen = enumerate(cycle(dataloader))
        for step, data in tqdm(gen, total=self._steps, desc=desc):
            if step >= self._steps:
                break
            if self._collate_fn is not None:
                data = self._collate_fn(data)
            executor.forward(inputs=data, hooks=hooks, output_names=output_names)

    @empty_ppq_cache
    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Optional[Iterable] = None,
        executor: Optional[BaseGraphExecutor] = None,
        calib_steps: int = 32,
        collate_fn: Optional[Callable[[Any], Any]] = None,
        **kwargs,
    ) -> None:
        assert dataloader is not None and executor is not None
        if collate_fn is not None:
            self._collate_fn = collate_fn
        if calib_steps is not None:
            self._steps = calib_steps
        # -------------------------------------------------
        # Override existing quantization configurations
        # -------------------------------------------------
        if self._override:
            for operation in graph.operations.values():
                if not isinstance(operation, QuantableOperation):
                    continue

                for config, var in operation.config_with_variable:
                    if (
                        not var.is_parameter
                        and config.state == QuantizationStates.ACTIVATED
                        and config.dominated_by == config
                    ):
                        config.state = QuantizationStates.INITIAL

        # build observer and hook for each quantable operation
        hooks = {}
        for op_name, operation in graph.operations.items():

            if not isinstance(operation, QuantableOperation):
                continue

            # override algorithm setting if necessary
            for config, var in operation.config_with_variable:
                if not var.is_parameter and self._method is not None:
                    config.observer_algorithm = self._method

            assert graph is executor.graph
            observer = OperationObserver(operation=operation, monitor_parameter=False)
            self._observers[op_name] = observer
            hooks[op_name] = observer.hook

        # ready for calibration
        # hook forward function, let observers take effects.
        self.calibrate(
            desc="Calibration Progress(Phase 1)",
            dataloader=dataloader,
            executor=executor,
            hooks=hooks,
        )

        # render calibration result.
        for _, observer in self._observers.items():
            assert isinstance(observer, OperationObserver)
            observer.render_quantization_config()
            observer.report()

        # -------------------------------------------------
        # There are some two-phase observer in ppq,
        # which means they have to be calibrated for a second time.
        #   see also: TorchHistObserver
        # -------------------------------------------------

        # remove one-phase observer from hook dict.
        pop_list = []
        for op_name, observer in self._observers.items():
            assert isinstance(observer, OperationObserver)
            if all(
                [
                    type(var_observer)
                    not in {OBSERVER_TABLE["kl"], OBSERVER_TABLE["mse"]}
                    for var_observer in observer.hook._observer_table.values()
                ]
            ):
                pop_list.append(op_name)

        for op_name in pop_list:
            self._observers.pop(op_name)
            hooks.pop(op_name)

        if len(hooks) > 0:
            # ready for calibration(Phase 2)
            # hook forward function, let observers take effects.
            self.calibrate(
                desc="Calibration Progress(Phase 2)",
                dataloader=dataloader,
                executor=executor,
                hooks=hooks,
            )

            # render calibration result for a second time.
            for _, observer in self._observers.items():
                assert isinstance(observer, OperationObserver)
                observer.render_quantization_config()
                observer.report()


@OPTIM_ALGORITHMS.register()
class IsotoneCalibrationPass(RuntimeCalibrationPass):
    """
    ## Isotone Calibration Pass(保序量化校准过程)

    在神经网络中，一些算子的输出并不需要保证总体的精确性，而只关注于最大最小值所在的位置，
    例如图像分类网络中，网络的输出通常是一个1000维的向量，用于表达图像属于特定类别的概率。
    为了保证分类的正确性，我们并不需要这个1000维的向量在量化后是整体准确的，只需要其中的最大值出现在正确的位置上。
    因此我们希望最大值与次大值之间相差至少半个 scale，并且次大值能够不被截断。

    因此传统的 min-max, percentile, kl 方法在这一情景中并不能得到最高的分类精度，
    保序量化是为了解决这一问题而设计的，在这一校准过程中，程序将网络输出变量的校准方式改写为 Isotone(保序校准)。
    默认设置下，该过程只对 softmax 算子的输出进行保序校准。对于其他情况，用户需要手动指定需要进行保序校准的变量名。

    保序量化需要设定一个分类轴，同样地以分类网络为例，其输出形为 [Batch, 1000]。
    分类操作将在数据的最后一维展开，因此需要设置保序轴为 -1。

    Algorithm:

        For softmax or sigmoid activations, usually we just need
        argmax(softmax(x)) == argmax(softmax(quant(x)))

        Inspired by this Property, Isotone Observer is designed to provide an
        order-preserving calibration method, which cares only about argmax(x)
        [or argmin(x)]

        To keep argmax(x) == argmax(quant(x)), we only need to distinguish the
        largest element and the second largert element with quantization

            let L1 represents the largest element of x,
            while L2 represents the second largest.

            For Symmetrical Quantization, We want:

                1. round(L1 / scale) - round(L2 / scale) > 0

                2. round(L2 / scale) < quant_max

            Hence that, we will have:

                1. scale < 2 * (L1 - L2)

                2. scale > L2 / (self._quant_cfg.quant_max - .5)

            For Asymmetircal Quantization, We want:

                1. round(L1 / scale) + offset - round(L2 / scale) - offset > 0

                2. round(L2 / scale) + offset < quant_max

            Hence that, we will have:

                1. scale < 2 * (L1 - L2)

                2. scale > L2 / (self._quant_cfg.quant_max - offset - .5)

        The best setting of scale, offset can be solved by PPQ Isotone observer.

        Time Complexity: O(nlogn)
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        axis: int = -1,
        calib_steps: int = 32,
    ) -> None:
        super().__init__(calib_steps=calib_steps)
        self.variables = variables
        self.axis = axis

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Optional[Iterable] = None,
        executor: Optional[BaseGraphExecutor] = None,
        calib_steps: int = 32,
        collate_fn: Optional[Callable[[Any], Any]] = None,
        **kwargs,
    ) -> None:
        if self.variables is None:
            for op in graph.operations.values():
                if op.type == "Softmax" and isinstance(op, QuantableOperation):
                    # had not been dominated.
                    if (
                        op.output_quant_config[0].dominated_by
                        == op.output_quant_config[0]
                    ):
                        op.output_quant_config[0].state = QuantizationStates.INITIAL
                        op.output_quant_config[0].observer_algorithm = "isotone"
                        op.output_quant_config[0].detail[
                            OBSERVER_ISOTONE_OBSERVER_AXIS
                        ] = op.attributes.get("axis", -1)

                        debug(
                            f"Calibration Method of Op {op.name} has been changed to "
                            f'Isotone[axis={op.attributes.get("axis", -1)}].'
                        )
        else:
            if not isinstance(self.variables, list):
                error("Isotone Calibration Pass needs a list of variable names.")
                raise TypeError
            for var in self.variables:
                if var not in graph.variables:
                    raise ValueError(f"Variable {var} not in current graph.")
                var = graph.variables[var]
                if isinstance(var, QuantableVariable):
                    source_config = var.source_op_config
                    assert source_config is not None
                    source_config.state = QuantizationStates.INITIAL
                    source_config.observer_algorithm = "isotone"
                    source_config.detail[OBSERVER_ISOTONE_OBSERVER_AXIS] = self.axis
                    debug(
                        f"Calibration Method of Variable {var.name} has been "
                        f"changed to Isotone[axis={self.axis}]."
                    )
        super().optimize(graph, dataloader, executor, calib_steps, collate_fn, **kwargs)
