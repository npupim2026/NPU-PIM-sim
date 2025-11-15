from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from logger import init_logger
from software.system_compiler.strategy import Strategy

if TYPE_CHECKING:
    from typing import List

    from software.compiler import CompileTechnique
    from software.system_compiler.system_info import SystemInfo


__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang", "Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("System Compiler", logging.INFO)


class SystemCompiler(object):
    def __init__(
        self,
        shared_compilers: List[CompileTechnique],
        multicore_compilers: List[Strategy],
        core_compilers: List[CompileTechnique],
        offchip_memory_compilers: List[Strategy],
        post_process_compilers: List[Strategy],
    ) -> None:
        self._shared_compilers = shared_compilers
        self._multicore_compilers = multicore_compilers
        self._core_compilers = core_compilers
        self._offchip_memory_compilers = offchip_memory_compilers
        self._post_process_compilers = post_process_compilers

    def compile(self, info: SystemInfo):
        logger.info("Start Pre-Compile Step")
        # Pre-Compilation --> Do not affect compile information
        for compiler in self._shared_compilers:
            info.shared_compile_info = compiler.compile(info.shared_compile_info)

        logger.info("Start System Compile Step")
        for compiler in self._multicore_compilers:
            info = compiler.compile(info)
        if not info.is_mapped:
            raise ValueError("Mapping stage must be required")

        logger.info("Start Core Compile Step")
        for compiler in self._core_compilers:
            for cinfo in info.core_info:
                cinfo.compile_info = compiler.compile(cinfo.compile_info)

        logger.info("Start Memory Compiliation")
        for compiler in self._offchip_memory_compilers:
            info = compiler.compile(info)

        logger.info("Start Post Processing")
        for compiler in self._post_process_compilers:
            info = compiler.compile(info)
            
        return info