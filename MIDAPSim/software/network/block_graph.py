from __future__ import annotations

import itertools
import logging
from copy import copy
from collections import OrderedDict
from functools import reduce
from typing import TYPE_CHECKING, Set

from graphviz import Digraph
from logger import init_logger
from software.network.graph import Graph
from software.network.layer import Layer
from software.network.model import ModelGraph
from software.network.types import OpType

from .block_node import BlockNode
from .layer_path import LayerPath

if TYPE_CHECKING:
    from typing import Callable, List, Tuple

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"

logger = init_logger("Block Graph", logging.INFO)


class BlockGraph(Graph):
    def __init__(self, name: str = None):
        self._name = name
        self._source: List[int] = []
        self._sink: List[int] = []
        self._id2block: OrderedDict[int, BlockNode] = OrderedDict()
        self._layer2block: OrderedDict[Layer, int] = OrderedDict()
        self._new_id = itertools.count()
        self._model = None
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def blocks(self) -> List[BlockNode]:
        return list(self._id2block.values())

    @property
    def inputs(self) -> List[BlockNode]:
        return [self._id2block[i] for i in self._source]

    def build(self, model: ModelGraph):
        self._model = model
        src_layers = model.inputs
        for src_layer in src_layers:
            paths = self._build_blocks_from_layer(src_layer)
            self._source.append(paths[0])
        logger.debug(f"Block build finished: srcs] {self.inputs}, {self._graph}")

    def _register_block(self, block: BlockNode):
        id = next(self._new_id)
        self._id2block[id] = block
        self._layer2block[block.source] = id
        return id

    def _build_blocks_from_layer(self, layer: Layer) -> List[int]:
        if layer in self._layer2block:
            return [self._layer2block[layer]]
        main_path = []
        sub_paths = []
        main_path_candidates = []
        outputs = layer.outputs
        logger.debug(f"Start to build blocks from layer {layer}, outputs {outputs}")
        next_layers = []
        for out in outputs:
            path, next_layer = self._get_sequential_layers(out)
            logger.debug(f"Find Path, Next Layer: {path}, {next_layer}")
            if not next_layer.outputs and len(next_layer.in_vtensors) == 1:
                path.append(next_layer)
                sub_paths.append(path)
            else:   # Main path candidate
                main_path_candidates.append((path, next_layer))
        if len(main_path_candidates) > 1:   # Multiple main path is not allowed; Separate each paths as new blocks
            for path, next_layer in main_path_candidates:
                if path:
                    next_layers.append(path[0])
                else:
                    next_layers.append(next_layer)
        elif len(main_path_candidates) == 1:
            main_path = main_path_candidates[0][0]
            next_layers.append(main_path_candidates[0][1])
        if main_path and any([l not in outputs and l.op.type == OpType.WeightedSum for l in self._model.get_next_node(layer)]):
            sub_paths.append([])
        block = BlockNode(layer, main_path, sub_paths)
        blk_id = self._register_block(block)
        if block.is_leaf:
            self.add_node(blk_id)
            self._sink.append(blk_id)
            return [blk_id]
        else:
            if not next_layers:
                raise RuntimeError("It must be a leaf node when no next_layers exist")
            ret_blks = []
            for next in next_layers:
                blocks = self._build_blocks_from_layer(next)
                self.add(blk_id, blocks[0])
                ret_blks = ret_blks + blocks
            return [blk_id] + blocks

    def _get_sequential_layers(self, layer: Layer) -> Tuple[List[Layer], Layer]:
        path = []
        # logger.debug(f"_get_sequential_layers: outputs = {layer.outputs}")
        while all([
            layer.outputs,
            len(self._model.get_next_node(layer)) == 1,
            len(layer.in_vtensors) == 1
        ]):
            path.append(layer)
            layer = layer.outputs[0]
        return path, layer

    def ordered_dfs(self) -> List[int]:
        model = self._model
        target_blks = self._source
        logger.debug(f"target_blks : {target_blks}")

        def _search(curr_block: int, next_targets=[], processed_blks=[], processed_layers=set()) -> List[int]:
            def _runnable(target: int):
                return set(
                    model.get_incoming_nodes(self._id2block[target].source)
                ).issubset(processed_layers)

            def _check_available(targets: List[int]) -> bool:
                if not targets:
                    return False
                return reduce(lambda acc, cur: acc or _runnable(cur), targets, False)

            processed_blks = copy(processed_blks)
            processed_layers = copy(processed_layers)
            while _runnable(curr_block):
                processed_blks.append(curr_block)
                if len(processed_blks) == len(self):
                    return processed_blks
                processed_layers.update(self._id2block[curr_block].layers)
                prioritized_target = list(self.get_next_node(curr_block))
                next_targets = list(filter(lambda b: b not in processed_blks, next_targets))
                #targets = prioritized_target if prioritized_target else next_targets
                targets = prioritized_target + next_targets

                if len(targets) == 1 and targets[0] not in processed_blks:
                    curr_block = targets[0]
                    continue
                else:
                    for target in targets:
                        result = _search(target, prioritized_target + next_targets, processed_blks, processed_layers)
                        if result:
                            return result
                    return []   # search failed
            return []

        for node in target_blks:
            processed_blks = _search(node, next_targets=target_blks)
            if len(processed_blks) == len(self):
                return processed_blks
        raise RuntimeError('BlockNode order search failed')
        return []

    @staticmethod
    def _draw(g: Digraph, block: BlockNode, id: int):
        with g.subgraph(name=f"cluster_{block.name}") as gb:
            gb.attr(label=str(id), labeljust='l')
            for layer in block.layers:
                gb.node(layer.name)
            for path in block.all_paths:
                LayerPath(path).draw(g)

    def draw(self, directory: str, filename: str = None):
        g = Digraph(self.name)
        func: Callable[[int], None] = lambda l: (
            self._draw(g, self._id2block[l], l)
        )
        self.traverse(func)
        g.render(directory=directory, filename=filename, format="png")

    def __len__(self) -> int:
        return len(self._id2block)

    def __getitem__(self, key):
        return self._id2block[key]
