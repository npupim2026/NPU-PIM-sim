from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from logger import init_logger
from software.compiler import CompileTechnique

if TYPE_CHECKING:
    from software.compiler.compile_info import CompileInfo
    from software.network.model import Layer


__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


logger = init_logger("Alignment", logging.INFO)


class AlignCompile(CompileTechnique):
    @classmethod
    def compile(cls, info: CompileInfo):
        model = info.model
        model.traverse(lambda l: cls.__align(l))
        logger.info("Done")
        return info

    @staticmethod
    def __align(layer: Layer):
        alignment = layer.op_info.alignment
        alignment.align(layer)
