from __future__ import annotations

import logging
from abc import ABC, abstractclassmethod
from typing import TYPE_CHECKING

from logger import init_logger

if TYPE_CHECKING:
    from software.system_compiler.system_info import SystemInfo

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


logger = init_logger("Compiler", logging.INFO)


class SystemCompileTechnique(ABC):
    @abstractclassmethod
    def compile(cls, info: SystemInfo) -> SystemInfo:
        pass
