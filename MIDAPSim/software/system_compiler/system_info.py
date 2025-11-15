from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import attr
from logger import init_logger
from software.system_compiler.dependency_info import DependencyInfo
from software.system_compiler.memory_info import MemoryInfo

if TYPE_CHECKING:
    from typing import List

    from software.compiler.compile_info import CompileInfo
    from software.system_compiler.core_info import CoreInfo

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"

logger = init_logger("System Info", logging.INFO)


@attr.s(slots=True, init=True)
class SystemInfo(object):
    ######
    #  XXX Recommandation for System Compilation
    #  1. Do not generate new tensor at the mapped model (cinfo.compile_info.model).
    #     Please do it first at the shared model and just allocate it to the dedicated core.
    #  2. Do not rename the layer/make a new layer at the mapped model.
    #     Please do it first at the shared model and just allocate it to the dedicated core.
    #     It makes easier to manage layer dependency information among cores.
    ######
    shared_compile_info: CompileInfo = (
        attr.ib()
    )  # Shared Compile Information (Whole Model, ...)
    core_info: List[CoreInfo] = attr.ib(default=attr.Factory(list))
    dependency_info: DependencyInfo = attr.ib(default=attr.Factory(DependencyInfo))
    memory_info: MemoryInfo = attr.ib(default=attr.Factory(MemoryInfo))

    @property
    def is_mapped(self):
        return len(self.core_info) > 0 and self.core_info[0].compile_info is not None

    def get_sim_memory_info(self, core_id):
        return self.memory_info.get_sim_memory_info(core_id)

    def get_address_dict(self, core_id):
        return self.memory_info.get_address_dict(core_id)

    def generate_prefetch_trace(self, core_idx):
        return self.memory_info.generate_prefetch_trace(core_idx, self.dependency_info)

    def get_core_layer_id_info(self, core_id):
        return self.dependency_info.get_core_layer_id_info(core_id)

    def get_core_dependency_info(self, core_id):
        return self.dependency_info.get_core_dependency_info(core_id)

    def __copy__(self):
        import copy

        new = SystemInfo()
        new.shared_compile_info = copy.copy(self.shared_compile_info)
        for cinfo in self.core_info:
            new.core_info.append(copy.copy(cinfo))
        new.dependency_info = copy.copy(self.dependency_info)
        new.memory_info = copy.copy(self.memory_info)
        return new
