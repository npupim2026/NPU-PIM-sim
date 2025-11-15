from __future__ import annotations

from typing import TYPE_CHECKING

from config import cfg

if TYPE_CHECKING:
    from software.network import LayerBlock

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class BlockStationary(object):
    def __init__(self, block: LayerBlock):
        self._block = block
        self._num_banks = cfg.MIDAP.FMEM.NUM

    def get_input_range(self):
        return range(0, self._num_banks)
