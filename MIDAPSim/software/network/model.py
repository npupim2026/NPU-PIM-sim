from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from graphviz import Digraph

from .graph import Graph
from .layer import Layer
from .tensor import Tensor
from .types import TensorType, OpType
from .virtual_tensor import VirtualTensor
from config import cfg

if TYPE_CHECKING:
    from typing import Callable, Dict, List

    from software.generic_op.operator_base import OperatorBase

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang", "Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class ModelGraph(Graph):
    def __init__(self, name: str):
        self._name = name
        self._source: List[Layer] = []
        self._input_tensors: List[Layer] = []
        self._str2tensor: OrderedDict[str, Tensor] = OrderedDict()
        self._str2vtensor: Dict[str, VirtualTensor] = {}
        self._str2layer: OrderedDict[str, Layer] = OrderedDict()
        super(ModelGraph, self).__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> List[Layer]:
        return self._source

    @property
    def inputs(self) -> List[Layer]:
        return self._input_tensors

    @property
    def layers(self) -> List[Layer]:
        return list(self._items)

    @property
    def tensors(self) -> List[Tensor]:
        return list(self._str2tensor.values())

    def build(self, op_dict: OrderedDict):
        op: OperatorBase
        for op in op_dict.values():
            layer = None
            if op.is_first_tensor and cfg.MIDAP.CONTROL_STRATEGY.FIRST_LAYER == 'EXCLUDE':
                vtensor = VirtualTensor.from_op(op, [])
                self._str2tensor[op.name] = vtensor.tensor
                self._str2vtensor[op.name] = vtensor
            else:
                layer = self._make_layer(op)
                # self._make_connection(layer)
                self.register_layer(layer)

    def _make_layer(self, op: OperatorBase) -> Layer:
        inputs = [self._str2layer[l] for l in op.input_layers if l in self._str2layer]
        in_vtensors = [self._str2vtensor[l] for l in op.input_layers]
        return Layer.from_op(op, {"layers": inputs, "vtensors": in_vtensors})

    def register_layer(self, layer: Layer):
        self.register_items(layer)
        self._make_connection(layer)

    def register_items(self, layer : Layer):
        layer.parent = self
        self._str2layer[layer.name] = layer
        for tensor in layer.get_tensors():
            self._str2tensor[tensor.name] = tensor
            self._str2tensor.move_to_end(tensor.name, last = True)
        for vtensor in layer.in_vtensors:
            self._str2tensor[vtensor.tensor.name] = vtensor.tensor
        self._str2tensor[layer.out_tensor.name] = layer.out_tensor
        self._str2tensor.move_to_end(layer.out_tensor.name, last = True)
        self._str2vtensor[layer.name] = layer.out_vtensor

    def _make_connection(self, layer: Layer):
        self.add_node(layer)
        if not layer.inputs:
            if not layer.dummy or cfg.MIDAP.CONTROL_STRATEGY.FIRST_LAYER == 'EXCLUDE':
                self.source.append(layer)
            self.inputs.append(layer)
        for prev in layer.inputs:
            if prev.dummy and not (prev.inputs or cfg.MIDAP.CONTROL_STRATEGY.FIRST_LAYER == 'EXCLUDE'): # a predecessor of a dummy input layer
                self.source.append(layer)
            self.add(prev, layer)

    def remove_connection(self, src: Layer, dst: Layer):
        # Warning: It removes only graph edge..
        import copy
        if src.out_vtensor in dst.in_vtensors and dst.op.type not in [OpType.MatMul, OpType.MatMulTrans, OpType.RoPE]:
            dst.in_vtensors.remove(src.out_vtensor)
            new_in_vtensor = copy.copy(src.out_vtensor)
            dst.in_vtensors.append(new_in_vtensor)
        src.outputs.remove(dst)
        dst.inputs.remove(src)
        if not dst.inputs:
            self.inputs.append(dst)
            if dst not in self.source:
                self.source.append(dst)

    @staticmethod
    def _draw(g: Digraph, layer: Layer):
        next: Layer
        for next in layer.parent.get_next_node(layer):
            if next in layer.outputs:
                g.edge(layer.name, next.name, label=layer.out_vtensor.name)
            else:
                g.edge(layer.name, next.name, label=layer.out_vtensor.name, style="dashed")

    def draw(self, directory: str):
        g = Digraph(self.name)
        func: Callable[[Layer], None] = lambda l: self._draw(g, l)
        self.traverse(func)
        g.render(directory=directory, format='png')

    def __len__(self) -> int:
        return len(self._str2layer)
