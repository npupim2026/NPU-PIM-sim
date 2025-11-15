from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from logger import init_logger

if TYPE_CHECKING:
    from software.system_compiler.system_info import SystemInfo


__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("Strategy", logging.INFO)


class Strategy(ABC):
    @abstractmethod
    def compile(self, info: SystemInfo) -> SystemInfo:
        return info
