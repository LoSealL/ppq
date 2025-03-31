"""
Copyright Wenyi Tang 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

# pylint: disable=missing-docstring

import onnx
from onnx.helper import make_graph, make_node, make_tensor_value_info

from mppq.api import load_onnx_graph
from mppq.dispatcher.scope import IgnoredScope, Subgraph


def _build_graph():
    r"""sigmoid    relu
         \       /
          \     /
           \   /
            add
             |
           split
            / \
           /   \
          /     \
       softmax  softmax
    """

    sigmoid = make_node("Sigmoid", ["input0"], ["sigmoid"], "my_name_is_sigmoid")
    relu = make_node("Relu", ["input1"], ["relu"], "my_name_is_relu")
    add = make_node("Add", ["sigmoid", "relu"], ["add"], "my_name_is_add")
    split = make_node(
        "Split",
        ["add"],
        ["softmax0", "softmax1"],
        "my_name_is_split",
        num_outputs=2,
        axis=1,
    )
    softmax0 = make_node("Softmax", ["softmax0"], ["output0"], "my_name_is_softmax0")
    softmax1 = make_node("Softmax", ["softmax1"], ["output1"], "my_name_is_softmax1")
    graph = make_graph(
        [sigmoid, relu, add, split, softmax0, softmax1],
        "test_graph",
        inputs=[
            make_tensor_value_info("input0", onnx.TensorProto.FLOAT, [1, 64]),
            make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [1, 64]),
        ],
        outputs=[
            make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 32]),
            make_tensor_value_info("output1", onnx.TensorProto.FLOAT, [1, 32]),
        ],
    )
    model = onnx.helper.make_model(
        graph,
        ir_version=onnx.IR_VERSION_2023_5_5,
        opset_imports=[onnx.helper.make_operatorsetid("", 19)],
    )
    onnx.checker.check_model(model)
    return load_onnx_graph(model)


def test_ignore_by_operation_name():
    graph = _build_graph()
    scope = IgnoredScope(operations=["my_name_is_add"])
    disp = scope.dispatch(graph)
    assert len(disp) == 1
    assert "my_name_is_add" in disp


def test_ignore_by_operation_type():
    graph = _build_graph()
    scope = IgnoredScope(types=["Softmax"])
    disp = scope.dispatch(graph)
    assert len(disp) == 2
    assert "my_name_is_softmax0" in disp
    assert "my_name_is_softmax1" in disp


def test_ignore_by_fuzzy_name():
    graph = _build_graph()
    scope = IgnoredScope(operations=["*sigmoid"])
    assert "my_name_is_sigmoid" in scope.dispatch(graph)

    scope = IgnoredScope(operations=["my_name_is_softmax?"])
    disp = scope.dispatch(graph)
    assert len(disp) == 2
    assert "my_name_is_softmax0" in disp
    assert "my_name_is_softmax1" in disp

    scope = IgnoredScope(operations=[r"re://my_name_is_\w{3,4}"])
    disp = scope.dispatch(graph)
    assert len(disp) == 2
    assert "my_name_is_relu" in disp
    assert "my_name_is_add" in disp


def test_ignore_by_subgraph():
    graph = _build_graph()
    scope = IgnoredScope(
        subgraphs=[
            Subgraph(
                ["my_name_is_sigmoid", "my_name_is_relu"],
                ["my_name_is_softmax0"],
            )
        ]
    )
    disp = scope.dispatch(graph)
    assert len(disp) == 5
    assert "my_name_is_softmax1" not in disp
