from __future__ import annotations

import logging
from software.network.types import TensorType
from typing import TYPE_CHECKING

from logger import init_logger
from software.compiler.compile_info import CompileInfo
from software.compiler.mapping_compile.policy import MappingCompilePolicy
from software.compiler.system_compile_technique import SystemCompileTechnique

if TYPE_CHECKING:
    from software.system_compiler.system_info import SystemInfo

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang", "Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


logger = init_logger("Mapping Compile", logging.DEBUG)


class MappingCompile(SystemCompileTechnique):
    policy: MappingCompilePolicy = None

    @classmethod
    def set_policy(cls, policy: MappingCompilePolicy):
        cls.policy = policy

    @classmethod
    def compile(cls, info: SystemInfo) -> SystemInfo:
        from copy import deepcopy

        cls.policy.compile()

        for model, core_info in zip(cls.policy.mapping, info.core_info):
            info.dependency_info.register_core_info(core_info.core_id, model)
            core_info.compile_info = CompileInfo(
                model, wmem_info_dict=deepcopy(info.shared_compile_info.wmem_info_dict)
            )
        info.core_info = cls.policy.core_info
        cls.update_tensor_info(info)
        return info

    @classmethod 
    def update_tensor_info(cls, info : SystemInfo) -> SystemInfo:
        # After Mapping, Tensor information must be renewed for the core compilation step
        tensors = info.shared_compile_info.model.tensors
        for tensor in tensors:
            shared_cnt = 0
            for core_info in info.core_info:
                if tensor in core_info.compile_info.model.tensors:
                    shared_cnt += 1
            if shared_cnt == 0: # FIXME: It should be done by interface
                logger.info(f"Tensor {tensor} is removed from the model [unused tensor]")
                del info.shared_compile_info.model._str2tensor[tensor.name]
            if shared_cnt > 1 or tensor.type in [TensorType.In, TensorType.Out]:
                tensor.shared = True
                logger.debug(f"Marking to shared tensor: {tensor}")
