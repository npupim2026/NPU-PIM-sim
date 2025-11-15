from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from logger import init_logger
from software.compiler.layer_compile import LayerInfo

if TYPE_CHECKING:
    from .compile_info import CompileInfo

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


logger = init_logger("Compile Info Analyzer", logging.INFO)


class CompileInfoAnalyzer(object):
    @classmethod
    def off_chip_access(cls, info: CompileInfo, start_idx: int):
        layers = info.layers[start_idx:]
        size: int = 0
        for layer in layers:
            if layer.dummy:
                continue
            size += cls._off_chip_access_feature(layer)
            size += cls._weight_load_size(layer)
        return size

    @staticmethod
    def _off_chip_access_feature(layer_info: LayerInfo) -> int:
        size = layer_info.load_access_size()
        size += layer_info.store_access_size()
        return size

    @staticmethod
    def _weight_load_size(layer_info: LayerInfo):
        return layer_info.weight_load_size()
