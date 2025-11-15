from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import attr
from logger import init_logger
from software.system_compiler.memory_info import LocalMemoryInfo

if TYPE_CHECKING:
    from software.compiler.compile_info import CompileInfo

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"

logger = init_logger("Address Info", logging.INFO)


@attr.s(slots=True, init=True)
class CoreInfo(object):
    core_id: int = attr.ib(default=-1)
    compile_info: CompileInfo = attr.ib(default=None)
    local_memory_info: LocalMemoryInfo = attr.ib(default=attr.Factory(LocalMemoryInfo))
