from typing import List

import torch

from mppq.data import DataType
from mppq.ir.base.processor import GraphCommandProcessor


class TrainableGraph(GraphCommandProcessor):
    """Trainable Graph offers a bunch of functions that provide training interfaces."""

    def parameters(self) -> List[torch.Tensor]:
        parameters = []
        for var in self.graph.variables.values():
            if var.is_parameter and DataType.to_torch(var.dtype) == torch.float:
                parameters.append(var.value)
        return parameters

    def zero_grad(self):
        for var in self.graph.variables.values():
            if var.is_parameter and DataType.to_torch(var.dtype) == torch.float:
                if var.value._grad is not None:
                    var.value._grad.zero_()

    def state_dict(self) -> dict:
        parameters = {}
        for var in self.graph.variables.values():
            if var.is_parameter and DataType.to_torch(var.dtype) == torch.float:
                parameters[var.name] = var.value
        return parameters

    @property
    def _acceptable_command_types(self):
        return []

    def process(self, command):
        return None
