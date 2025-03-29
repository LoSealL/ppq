from collections import defaultdict
from contextlib import suppress
from typing import Callable, Dict, Iterable, List, Literal, Optional, Union

import torch
from tqdm import tqdm

from mppq.data import convert_any_to_numpy
from mppq.executor import TorchExecutor
from mppq.executor.base import BaseGraphExecutor
from mppq.ir.base.graph import BaseGraph
from mppq.ir.base.quantize import QuantableOperation
from mppq.ir.deploy import QuantableGraph
from mppq.quantization.analyse.util import MeasurePrinter, MeasureRecorder
from mppq.utils.fetch import tensor_random_fetch


def layerwise_error_analyse(  # noqa: C901
    executor: BaseGraphExecutor,
    dataloader: Iterable,
    interested_outputs: Optional[Union[str, List[str]]] = None,
    collate_fn: Optional[Callable] = None,
    method: Literal["cosine", "snr", "mse"] = "snr",
    reduce: Literal["mean", "sum"] = "mean",
    steps: int = 8,
    verbose: bool = True,
) -> Dict[str, float]:
    """Measure the quantization error of each operation A dictionary contains
    output differences for all operation will be returned as a result.

        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}

    if verbose is set as True, this function will display error report at last.

    The key of the dictionary is an operation name while the value of corresponding key
        is the difference between quantized output and float output of this operation.

    Result {'operation name 1': 0.933} means quantizing operation 1
        will generates 0.933 quantization error to output variable

    ATTENTION: Output difference is measured at operation-level.

    Args:
        graph (BaseGraph):
            A fully quantized graph instance.

        dataloader (Iterator):
            Test dataloader, this function will measure quantization error based on
            given data.

        collate_fn (Callable, optional):
            An data preprocessing function provided by user to convert data from
            dataloader towards executable format. If set as None, then no action will
            be taken during preprocessing.

        method (str, optional):
            A string indicates a measurement to calculate the difference of quantized
            output and fp32 one. 'cosine', 'snr', and 'mse' is supported in PPQ for now.

        steps (Int, optional)
            computation steps.

        interested_outputs (Union[str, List[str]] = None)
            a list contains your interested output variables. if set as None, all graph
            output variables will be measured via this function.

    Returns:
        A dictionary contains output differences for all operation will be returned
        from this function.

        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}
    """

    graph = executor._graph  # pylint: disable=protected-access
    assert graph is not None
    if interested_outputs is None:
        interested_outputs = [name for name in graph.outputs]

    if isinstance(interested_outputs, str):
        interested_outputs = [interested_outputs]

    # find all quantable operations.
    quantable_operations = []
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.dequantize()

            # we only need reports from computing op.
            if operation.is_computing_op:
                quantable_operations.append(operation)

    # dequantize all operations, create recorder for each operation
    recorders = {}
    for operation in quantable_operations:
        if isinstance(operation, QuantableOperation):
            recorders[operation.name] = MeasureRecorder(
                measurement=method, reduce=reduce
            )

    # run for each quantable operations:
    for operation in tqdm(
        quantable_operations, desc="Analysing Layerwise quantization error:"
    ):
        assert isinstance(operation, QuantableOperation)
        recorder = recorders[operation.name]
        assert isinstance(recorder, MeasureRecorder)

        for idx, batch in enumerate(dataloader):
            if collate_fn is not None:
                batch = collate_fn(batch)
            fp_outputs = executor.forward(inputs=batch, output_names=interested_outputs)

            # manually override quantization state
            operation.restore_quantize_state()
            qt_outputs = executor.forward(inputs=batch, output_names=interested_outputs)

            for fp_output, qt_output in zip(fp_outputs, qt_outputs):
                recorder.update(y_pred=qt_output, y_real=fp_output)

            # manually override quantization state
            operation.dequantize()
            if idx >= steps:
                break

    # restore quantization states
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.restore_quantize_state()

    results: Dict[str, float] = {}
    for operation in quantable_operations:
        assert isinstance(operation, QuantableOperation)
        results[operation.name] = recorders[operation.name].measure

    if verbose:
        method_str = "MEASUREMENT"
        if method == "snr":
            method_str = "NOISE:SIGNAL POWER RATIO"
        if method == "cosine":
            method_str = "COSINE SIMILARITY"
        if method == "mse":
            method_str = f"{reduce.upper()} SQUARE ERROR"
        MeasurePrinter(
            results,
            order="large_to_small",
            measure=method_str,
            percentage=method in {"snr", "cosine"},
        ).print()
    return results


def variable_analyse(
    graph: BaseGraph,
    dataloader: Iterable,
    interested_outputs: Union[str, List[str]],
    collate_fn: Optional[Callable] = None,
    running_device="cuda",
    samples_per_step: int = 65536,
    steps: int = 8,
    dequantize: bool = False,
):
    if isinstance(interested_outputs, str):
        interested_outputs = [interested_outputs]

    quant_graph = QuantableGraph(graph)

    executor = TorchExecutor(graph=graph, device=running_device)
    if dequantize:
        quant_graph.dequantize_graph()

    data_collector = defaultdict(list)
    for idx, batch in enumerate(dataloader):
        if collate_fn is not None:
            batch = collate_fn(batch)
        fp_outputs = executor.forward(inputs=batch, output_names=interested_outputs)
        for output, output_name in zip(fp_outputs, interested_outputs):
            data_collector[output_name].append(
                tensor_random_fetch(
                    tensor=output, num_of_fetches=samples_per_step
                ).unsqueeze(0)
            )
        if idx >= steps:
            break

    for name in interested_outputs:
        # pylint: disable=import-outside-toplevel
        from matplotlib import pyplot as plt

        tensor = torch.cat(data_collector[name]).flatten()
        tensor = convert_any_to_numpy(tensor, False)

        plt.figure(figsize=(12, 8))
        plt.title(f"Histogram Result of Variable {name}:")
        plt.hist(tensor, bins=64)
        plt.show()

    if dequantize:
        quant_graph.restore_quantize_state()


def parameter_analyse(graph: BaseGraph):
    ranges, stds, means = {}, {}, {}
    for operation in graph.operations.values():
        for var in operation.parameters:
            value = var.value
            assert isinstance(value, torch.Tensor)
            if value.numel() <= 1:
                continue

            _min, _max, _std, _mean = 0, 0, 0, 0
            with suppress(Exception):
                _min = value.min().item()
                _max = value.max().item()
                _std = value.std().item()
                _mean = value.mean().item()

            ranges[f"{var.name}[{operation.name}]"] = _max - _min
            stds[f"{var.name}[{operation.name}]"] = _std
            means[f"{var.name}[{operation.name}]"] = abs(_mean)

    MeasurePrinter(ranges, order="large_to_small", measure="Value Range").print()
    MeasurePrinter(stds, order="large_to_small", measure="Value Std").print()
    MeasurePrinter(means, order="large_to_small", measure="Value Mean(Abs)").print()
