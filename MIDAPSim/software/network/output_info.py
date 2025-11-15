from __future__ import annotations

from typing import TYPE_CHECKING

from .virtual_tensor import VirtualTensor

if TYPE_CHECKING:
    from typing import List

    from software.generic_op.operator_base import OperatorBase

    from .layer import Layer


class OutputInfo(object):
    __slots__ = ("__vtensor", "__layers")

    def __init__(self, vtensor: VirtualTensor, layers: List[Layer]):
        self.__vtensor = vtensor
        self.__layers = layers

    @classmethod
    def from_op(
        cls, op: OperatorBase, input_vtensors: List[VirtualTensor]
    ) -> OutputInfo:
        vtensor = VirtualTensor.from_op(op, input_vtensors)
        layers = []
        return cls(vtensor, layers)

    @property
    def num(self) -> int:
        return len(self.__layers)

    @property
    def layers(self) -> List[Layer]:
        return self.__layers

    @property
    def vtensor(self) -> VirtualTensor:
        return self.__vtensor

    def add_output(self, layer: Layer):
        self.__layers.append(layer)
