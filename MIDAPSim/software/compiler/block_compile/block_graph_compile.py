from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import cfg
from logger import init_logger
from software.compiler import CompileTechnique
from software.compiler.model_transformer import ModelTransformer, PruningHeuristic
from software.network.block_graph import BlockGraph
from software.network.block_node import BlockNode
from software.compiler.layer_compile.behavior import Behavior
from software.network.op_info import OpType

if TYPE_CHECKING:
    from typing import List

    from software.compiler.compile_info import CompileInfo
    from software.network import ModelGraph

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang", "Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("Block Graph Compile", logging.INFO)


class BlockGraphCompile(CompileTechnique):
    model_transformer: ModelTransformer = PruningHeuristic()

    @classmethod
    def compile(cls, info: CompileInfo):
        model = info.model = cls.model_transformer.transform(info.model)
        logger.info("=================== BlockCompile ===================")
        blk_graph = cls.__build_blk(model)
        info = cls.__determine_order_stationary(info, blk_graph)
        info = cls.__postprocess_adjust_behavior(info)
        logger.debug(f"\n{[l.layer.name for l in info.layers]}")
        logger.info("====================================================")
        return info

    @classmethod
    def __build_blk(cls, model: ModelGraph) -> BlockGraph:
        blk_graph = BlockGraph(model.name + "_blkgraph")
        blk_graph.build(model)
        logger.info(f"Num Blocks: {len(blk_graph)}")
        logger.info(f"Blocks    :")
        for id, block in blk_graph._id2block.items():
            logger.info(f"{id}: {block}")
        if cfg.MODEL.GENERATE_BLOCK_DOT:
            cls.__draw_blks(model, blk_graph)
        return blk_graph

    @staticmethod
    def __draw_blks(model: ModelGraph, blk_graph: BlockGraph):
        import os

        dot_directory = cfg.MODEL.DOT_DIRECTORY
        dot_prefix = f"{model.name}_graph"
        blk_graph.draw(os.path.join(dot_directory, dot_prefix))

    @classmethod
    def __determine_order_stationary(
        cls, info: CompileInfo, blk: BlockGraph
    ) -> CompileInfo:
        blocks = blk.ordered_dfs()
        blocks: List[BlockNode] = [blk._id2block[i] for i in blocks]
        for v in blocks:
            info = cls.__generate_block_info(info, v)
            logger.info(
                f"{v}: {[(l.name, info.get_layer_info(l).stationary) for l in v.layers]}"
            )
        return info

    @staticmethod
    def __generate_block_info(info: CompileInfo, block: BlockNode) -> CompileInfo:
        from .block_node_optimizer import BlockNodeOptimizer

        optimizer = BlockNodeOptimizer(block, info)
        return optimizer.optimize_block()

    @classmethod
    def __postprocess_adjust_behavior(cls, info: CompileInfo):
        layer_info_list = [layer_info for layer_info in info.layer_dict.values()]
        for i, layer_info in enumerate(reversed(layer_info_list[2:])):
            first_load = []
            for behavior in layer_info.behavior:
                if behavior.is_load():
                    first_load.append(behavior)
                else:
                    break
            for j in range(2, len(layer_info_list)-i):
                prev_layer = layer_info_list[-i-j]
                if prev_layer.op.type in [OpType.Concat, OpType.HostProcess] or not prev_layer.dummy:
                    break
                if i + j == len(layer_info_list) - 1:   # Dummy first layer?
                    return info
            if prev_layer.op.type in [OpType.Concat, OpType.HostProcess]:
                continue
            # First LOAD behavior
            insert_idx = 0
            for n in range(len(prev_layer.behavior)):
                if prev_layer.behavior[n].is_load():
                    insert_idx = n + 1
                else:
                    break
            for behavior in first_load:
                behav_in_map = next(filter(lambda m: m.data == behavior.data[0], layer_info.mapping.input))
                prev_out_pos = [item for item in filter(lambda m: m.bank == behav_in_map.bank, prev_layer.mapping.output)]
                if prev_out_pos:    # The bank is already occupied at the end of the prev. layer
                    break
                prev_in_pos = [item for item in filter(lambda m: m.bank == behav_in_map.bank, prev_layer.mapping.input)]
                insert_idx = max([insert_idx] + [idx + 1 for idx in filter(lambda i: prev_layer.behavior[i].is_process()
                                                 and any(m.data in prev_layer.behavior[i].data for m in prev_in_pos), range(insert_idx, len(prev_layer.behavior)))])
                if insert_idx == len(prev_layer.behavior): # Prefetching this data is meaningless and may be harmful (WMEM prefetch should be prioritized)
                    break
                layer_info.behavior.remove(behavior)
                if prev_layer.out_tensor == behav_in_map.data.tensor and (not prev_in_pos or prev_layer.input2output(prev_layer.behavior[insert_idx].data)[-1].last_x <= behavior.data[-1].pivot):
                    try:
                        prev_layer.mapping.output.remove(next(filter(lambda m: m.data == behav_in_map.data, prev_layer.mapping.output)))
                    except StopIteration:
                        pass
                    prev_layer.mapping.output.append(behav_in_map)
                else:
                    proc_behav = [item for item in filter(lambda b: b.is_process() and behav_in_map.data in b.data, prev_layer.behavior[insert_idx:])]
                    if proc_behav:
                        try:
                            prev_layer.behavior.remove(next(filter(lambda b: b.is_load() and b.data[0] == behav_in_map.data, prev_layer.behavior)))
                        except StopIteration:
                            pass
                        else:
                            insert_idx -= 1
                        prev_layer.mapping.input.remove(next(filter(lambda m: m.data == behav_in_map.data, prev_layer.mapping.input)))
                    prev_layer.behavior.insert(insert_idx, behavior)
                    insert_idx += 1
                    prev_layer.mapping.input.append(behav_in_map)
            prev_layer.mapping.input = sorted(prev_layer.mapping.input, key=(lambda m: (m.data.pivot, prev_layer.ordered_input_tensors.index(m.data.tensor) if m.data.tensor in prev_layer.ordered_input_tensors else -1)))
        return info
