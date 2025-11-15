from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from logger import init_logger
from software.compiler.system_compile_technique import SystemCompileTechnique
from software.compiler.wmem_precompile import WMEMPreCompile
from software.network.model import ModelGraph
from software.system_compiler.dependency_info import DependencyInfo
from software.network.types import OpType

if TYPE_CHECKING:
    from typing import List

    from software.compiler.model_transformer import ModelTransformer
    from software.network.layer import Layer
    from software.system_compiler.system_info import SystemInfo

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang", "Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"

logger = init_logger("Graph Compile", logging.DEBUG)


class GraphCompile(SystemCompileTechnique):
    model_transformer: ModelTransformer = None

    @classmethod
    def set_transformer(cls, transform):
        cls.model_transformer = transform

    @classmethod
    def _get_dependency_info(cls, model: ModelGraph):
        dependency_info = DependencyInfo()

        def _add_dependency(src: Layer, out_layers: List[Layer]):
            if src.virtual or src.reduction or src.op.type == OpType.Dummy: # FIXME : Temporal, Reduction
                return
            for dst in out_layers:
                if dst.virtual or dst.reduction or dst.op.type == OpType.Dummy:
                    _add_dependency(src, model.get_next_node(dst))
                else:
                    dependency_info.add_global_dependency(dst.name, src.name)

        model.traverse(lambda l: _add_dependency(l, model.get_next_node(l)))
        return dependency_info

    @classmethod
    def compile(cls, info: SystemInfo) -> SystemInfo:
        model = info.shared_compile_info.model = (
            cls.model_transformer.transform(info.shared_compile_info.model)
            if cls.model_transformer
            else info.shared_compile_info.model
        )

        info.dependency_info = cls._get_dependency_info(model)
        info.shared_compile_info = WMEMPreCompile.compile(info.shared_compile_info)
        logger.info("Graph Compile Finished.")
        return info
