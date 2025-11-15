from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .intermediate_info import IntermediateInfo

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang", "Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class LayerCompliePolicy(ABC):
    @abstractmethod
    def search(self, info: IntermediateInfo, force_process: bool = False):
        pass
