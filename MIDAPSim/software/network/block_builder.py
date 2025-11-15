from __future__ import annotations

from collections import defaultdict
from functools import reduce
from sys import maxsize
from typing import TYPE_CHECKING

import numpy as np
from orderedset import OrderedSet

from .layer_block import LayerBlock

if TYPE_CHECKING:
    from typing import List, Set, Tuple

    from .layer import Layer
    from .model import ModelGraph

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class BlockBuilder(object):
    _layer2level = {}
    _model = None

    @classmethod
    def make_blkpath(cls, model: ModelGraph) -> Set[Layer]:
        cls._model = model
        cls._set_level()
        return cls._make_paths()

    @classmethod
    def _set_level(cls):
        nexts = cls._model.inputs
        level = 0
        visits = defaultdict(int, {layer: layer.num_inputs - 1 for layer in cls._model})
        while nexts:
            curs = nexts
            nexts = []
            for layer in curs:
                if visits[layer] > 0:
                    visits[layer] -= 1
                    continue
                cls._layer2level[layer] = level
                nexts.extend(layer.outputs)
            level += 1

    @classmethod
    def _make_paths(cls) -> Set[Layer]:
        paths = OrderedSet(cls._model.inputs)
        layer: Layer = paths[0]
        while layer.num_outputs != 0:
            if layer.num_outputs == 1:
                out_layers = layer.outputs
                paths.add(out_layers[0])
                layer = out_layers[0]
            elif layer.num_outputs > 1:
                paths = paths | cls._make_block(layer)
                layer = paths[-1]
        return paths

    @classmethod
    def _make_block(cls, layer: Layer) -> Set:
        nexts = [OrderedSet([layer, l]) for l in layer.outputs]
        level = reduce(lambda x, y: min(x, cls._layer2level[y[-1]]), [maxsize] + nexts)
        while len(nexts) > 1:
            nexts = cls._extend_paths(nexts, level)
            level += 1
            paths2merge = cls._find_block(nexts, level)
            if paths2merge:
                nexts = cls._merge_paths(nexts, paths2merge)
        return nexts[0]

    @classmethod
    def _merge_paths(cls, paths: List[Set], paths2merge: List[list]) -> List[Set]:
        for pl in reversed(paths2merge):
            merged_path = OrderedSet()
            blk_info = cls._get_block_info(pl, merged_path)
            blk = LayerBlock(*blk_info)
            merged_path.update([blk_info[0], blk, blk_info[1]])
            paths = [p for p in paths if p not in pl]
            paths.append(merged_path)
        return paths

    @classmethod
    def _get_block_info(
        cls, paths: List[Set], merged_path: OrderedSet
    ) -> Tuple[Layer, Layer, List[Layer]]:
        min_length = reduce(lambda x, y: min(x, len(y)), [maxsize] + paths)
        assert min_length > 0 and min_length != maxsize
        np_paths = np.asarray([p[:min_length] for p in paths])
        source, blk_paths = None, None
        for idx in range(min_length):
            idx_layers = np_paths[:, idx]
            if np.all(idx_layers == idx_layers[0]):
                merged_path.add(idx_layers[0])
                source = idx_layers[0]
            else:
                blk_paths = [p[idx:-1] for p in paths]
                break
        return source, paths[0][-1], blk_paths

    @classmethod
    def _find_block(cls, paths: List[Set], level: int) -> List[List[Set]]:
        merged = set()
        paths2merge = []
        for idx, p1 in enumerate(paths):
            if cls._layer2level[p1[-1]] > level or p1[-1] in merged:
                continue
            _candidate = [p2 for p2 in paths[idx + 1 :] if p2[-1] == p1[-1]]
            if _candidate:
                paths2merge.append(_candidate)
                paths2merge[-1].append(p1)
                merged.add(p1[-1])
        return paths2merge

    @classmethod
    def _extend_paths(cls, paths: List[Set], level: int) -> List[Set]:
        from copy import copy

        _paths = copy(paths)
        for idx, path in enumerate(_paths):
            cur = path[-1]
            if cls._layer2level[cur] == level:
                paths[idx] = cls._extend_layer(cur, paths[idx])
            elif cls._layer2level[cur] < level:
                raise ValueError(
                    f"Do not support output layer in the middle of network. (Layer: {cur.name})"
                )
        return paths

    @classmethod
    def _extend_layer(cls, layer: Layer, path: Set):
        if layer.num_outputs > 1:
            blk = cls._make_block(layer)
            path = path | blk
        elif layer.outputs:
            path.add(*layer.outputs)
        return path
