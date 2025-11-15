from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from software.compiler.layer_compile.behavior import Action, SystemBehavior
from software.compiler.layer_compile.layer_info import LayerInfo
from software.network.tensor import Tensor
from software.network.types import OpType, TensorType
from software.system_compiler.core_info import CoreInfo

from .strategy import Strategy

from logger import init_logger

if TYPE_CHECKING:
    from software.system_compiler.system_info import SystemInfo


__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("PostProcess", logging.INFO)


class PostProcess(Strategy):
    def _initialize(self, info: SystemInfo) -> None:
        self._shared_info = info.shared_compile_info
        self._model = info.shared_compile_info.model
        self._dep_info = info.dependency_info
        self._core_info = info.core_info
        self._memory_info = info.memory_info


    def compile(self, info: SystemInfo) -> SystemInfo:
        self._initialize(info)
        self._run()
        info.core_info = self._core_info
        return info

    def _run(self) -> None:
        pass

class Prototype(PostProcess):
    def _run(self) -> None:
        for cinfo in self._core_info:
            self._post_proc_core(cinfo)

    def _post_proc_core(self, core_info : CoreInfo):
        core_id = core_info.core_id
        id_dict = self._dep_info.get_core_layer_id_info(core_id)
        dep_info = self._dep_info.get_core_dependency_info(core_id)
        for l in core_info.compile_info.layers:
            name = l.name
            action = l.behavior
            # Sync
            action = action + [SystemBehavior(Action.SYNC, id_dict[name])]
            # Wait
            for sync_id in dep_info[name]:
                action = [SystemBehavior(Action.WAIT, sync_id)] + action
            l.behavior = action

