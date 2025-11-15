from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

    from .layer import Layer
    from .virtual_tensor import VirtualTensor

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class InputInfo(object):
    __slots__ = ("__layers", "__vtensors", "__is_network_input")

    def __init__(self, layers, vtensors):
        self.__layers: List[Layer] = layers
        self.__vtensors: List[VirtualTensor] = vtensors
        self.__is_network_input = any([t.is_input_tensor for t in self.__vtensors])

    @property
    def num(self) -> int:
        return len(self.__layers)

    @property
    def layers(self) -> List[Layer]:
        return self.__layers

    @property
    def vtensors(self) -> List[VirtualTensor]:
        return self.__vtensors

    @property
    def is_first_layer(self):
        return self.__is_network_input

    def set_prev(self, layer: Layer) -> None:
        self._set_prev_tensor(layer.out_tensor)
        self._set_prev_layer(layer)

    def _set_prev_layer(self, layer: Layer) -> None:
        for idx, l in enumerate(self.__layers):
            if l == layer:
                break

        self.__layers[0], self.__layers[idx] = (
            self.__layers[idx],
            self.__layers[0],
        )

    def _set_prev_tensor(self, tensor) -> None:
        for idx, vt in enumerate(self.__vtensors):
            if vt.tensor == tensor:
                break

        self.__vtensors[0], self.__vtensors[idx] = (
            self.__vtensors[idx],
            self.__vtensors[0],
        )
