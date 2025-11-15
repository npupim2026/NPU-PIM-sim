from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import attr
from logger import init_logger
from software.compiler.wmem_info import WMEMInfo

from .layer_compile.fmem_info import FMEMInfo
from .layer_compile.layer_info import LayerInfo
from software.network.types import OpType

if TYPE_CHECKING:
    from typing import List

    from software.network import Layer, ModelGraph


__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"

logger = init_logger("Compile Info", logging.INFO)


@attr.s(slots=True, init=True)
class CompileInfo(object):
    model: ModelGraph = attr.ib()
    layer_dict: OrderedDict[Layer, LayerInfo] = attr.ib(
        default=attr.Factory(OrderedDict)
    )
    num_compiled: int = attr.ib(default=0)
    fmem_info: FMEMInfo = attr.ib(default=attr.Factory(FMEMInfo))
    wmem_info_dict: OrderedDict[str, WMEMInfo] = attr.ib(
        default=attr.Factory(OrderedDict)
    )

    @property
    def layers(self) -> List[LayerInfo]:
        return list(self.layer_dict.values())

    def __copy__(self):
        from copy import copy

        new = CompileInfo(
            model=self.model,
            num_compiled=self.num_compiled,
            wmem_info_dict=self.wmem_info_dict,
        )
        new.layer_dict.update(self.layer_dict)
        new.fmem_info = copy(self.fmem_info)
        return new

    def update_layer_dict(self, ldict: OrderedDict[Layer, LayerInfo]):
        self.layer_dict.update(ldict)

    def is_layer_added(self, layer: Layer) -> bool:
        return layer in self.layer_dict

    def get_layer_info(self, layer: Layer) -> LayerInfo:
        return self.layer_dict[layer]

    def append_layer(self, layer: Layer):
        if all([l in self.layer_dict for l in layer.inputs]):
            info = LayerInfo.from_layer(layer)
            actual_outputs = sum((l.outputs if l.virtual else [l] for l in layer.outputs), [])
            if not actual_outputs:
                info.stationary.output = 0
            elif all([l.reduction for l in actual_outputs]):
                info.stationary.output = 0
            self.layer_dict[layer] = info
