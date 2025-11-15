from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import cfg
from logger import init_logger
from software.compiler import CompileTechnique
from software.compiler.layer_compile import LayerCompile
from software.network import LayerBlock

if TYPE_CHECKING:
    from typing import List

    from software.compiler.compile_info import CompileInfo
    from software.network import Layer, ModelGraph

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


logger = init_logger("Block Compile", logging.DEBUG)


class BlockCompile(CompileTechnique):
    @classmethod
    def compile(cls, info: CompileInfo):
        model = info.model
        logger.info("=================== BlockCompile ===================")
        path = cls.__build_blk(model)
        info = cls.__determine_order_stationary(info, path)
        logger.debug(f"\n{[l.layer.name for l in info.layers]}")
        logger.info("====================================================")
        return info

    @classmethod
    def __build_blk(cls, model: ModelGraph) -> List[LayerBlock]:
        from software.network.block_builder import BlockBuilder

        path = BlockBuilder.make_blkpath(model)
        blks = [b for b in path if isinstance(b, LayerBlock)]
        logger.info(f"Num Blocks: {len(blks)}")
        logger.info(f"Blocks    : {blks}")
        if cfg.MODEL.GENERATE_BLOCK_DOT:
            cls.__draw_blks(model, blks)
        return path

    @staticmethod
    def __draw_blks(model: ModelGraph, blks: List[LayerBlock]):
        LayerBlock.dot_directory = cfg.MODEL.DOT_DIRECTORY
        LayerBlock.dot_prefix = f"{model.name}_"
        for b in blks:
            b.draw()

    @classmethod
    def __determine_order_stationary(
        cls, info: CompileInfo, path: List[Layer | LayerBlock]
    ) -> CompileInfo:
        for v in path:
            if isinstance(v, LayerBlock):
                info = cls.__generate_block_info(info, v)
                logger.info(
                    f"{v}: {[(l.name, info.get_layer_info(l).stationary) for l in v.source.outputs]}"
                )
            else:
                cls.__generate_layer_info(info, v)
        return info

    @staticmethod
    def __generate_layer_info(info: CompileInfo, layer: Layer):
        if info.is_layer_added(layer):
            return
        info.append_layer(layer)
        if not LayerCompile.compile(info):
            raise RuntimeError(f"Could not compile. FMEM size may be too small for this model. ({layer})")

    @staticmethod
    def __generate_block_info(info: CompileInfo, block: LayerBlock) -> CompileInfo:
        from .block_optimizer import BlockOptimizer

        optimizer = BlockOptimizer(block, info)
        return optimizer.explore_order_stationary()
