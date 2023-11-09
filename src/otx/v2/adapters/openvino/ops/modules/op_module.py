"""Operation module for otx.v2.adapters.openvino.ops.modeuls."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Dict, List, Optional, Union

import torch

from otx.v2.adapters.openvino.ops.op import Attribute, Operation
from otx.v2.adapters.openvino.ops.utils import convert_op_to_torch


class OperationModule(torch.nn.Module):
    """OperationModule class."""

    def __init__(
        self,
        op_v: Operation,
        dependent_ops: Union[List[Operation], Dict[str, Optional[Operation]]],
    ) -> None:
        super().__init__()

        self.op_v = op_v
        self._dependent_ops = torch.nn.ModuleDict()

        spec = inspect.getfullargspec(op_v.forward)
        kwargs = spec.args[1:]

        self._dependents_with_defaults = []
        if spec.defaults:
            self._dependents_with_defaults = spec.args[-len(spec.defaults) :]

        if isinstance(dependent_ops, list):
            for op_, kwarg in zip(dependent_ops, kwargs):
                self._dependent_ops[kwarg] = op_
        elif isinstance(dependent_ops, dict):
            for kwarg in kwargs:
                self._dependent_ops[kwarg] = dependent_ops[kwarg]
        else:
            raise NotImplementedError

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Operationmodule's forward function."""
        inputs = {k: v() if v is not None else None for k, v in self._dependent_ops.items()}

        if args:
            empty_input_keys = [k for k, v in self._dependent_ops.items() if v is None]
            for key, val in zip(empty_input_keys, args):
                inputs[key] = val
        if kwargs:
            for key, val in kwargs.items():
                if inputs[key] is not None:
                    raise ValueError(f"duplicated key {key}")
                inputs[key] = val

        return self.op_v(**inputs)

    @property
    def type(self) -> str:
        """Operationmodule's type property."""
        return self.op_v.type

    @property
    def version(self) -> str:
        """Operationmodule's version property."""
        return self.op_v.version

    @property
    def name(self) -> str:
        """Operationmodule's name property."""
        return self.op_v.name

    @property
    def shape(self) -> Optional[tuple]:
        """Operationmodule's shape property."""
        return self.op_v.shape

    @property
    def attrs(self) -> Attribute:
        """Operationmodule's attrs property."""
        return self.op_v.attrs


def convert_op_to_torch_module(target_op: Operation) -> OperationModule:
    """Convert op Node to torch module."""
    dependent_modules = []
    for in_port in target_op.inputs():
        out_port = in_port.get_source_output()
        parent = out_port.get_node()

        parent_type = parent.get_type_name()
        if parent_type == "Constant":
            dependent_modules.append(convert_op_to_torch(parent))
        else:
            dependent_modules.append(None)
    module = convert_op_to_torch(target_op)
    module = OperationModule(module, dependent_modules)
    return module