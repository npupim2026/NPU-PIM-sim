from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING

import attr
from software.network import LayerBlock
from software.network.block_node import BlockNode
from software.network.layer_path import LayerPath
from software.network.op_info import OpType

if TYPE_CHECKING:
    from typing import Generator, List, Tuple

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


@attr.s(slots=True)
class PathOrder(object):
    paths: Tuple[LayerPath] = attr.ib()

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return iter(self.paths)

    def __getitem__(self, idx: int):
        return self.paths[idx]

    def get_last_layer(self):
        return self.paths[-1].get_last_layer()


class PathOrderSpace(object):
    __slots__ = ("_fixed_head", "_fixed_tail", "_explore")

    def __init__(self, fixed_head, fixed_tail, explore):
        self._fixed_head = fixed_head
        self._fixed_tail = fixed_tail
        self._explore = explore

    @classmethod
    def from_layer_block(cls, block: LayerBlock):
        paths = block.paths
        sink_info = block.sink.op_info
        if sink_info.type == OpType.Concat:
            fh, ft, explore = cls.__x_concat_search_space(paths)
        else:
            fh, ft, explore = cls.__others_search_space(paths)
        return cls(fh, ft, explore)

    @classmethod
    def from_block_node(cls, block: BlockNode):
        fh, ft, explore = cls.__block_node_search_space(block)
        return cls(fh, ft, explore)

    @classmethod
    def __x_concat_search_space(self, paths: List[LayerPath]):
        return paths, [], []

    @classmethod
    def __others_search_space(self, paths: List[LayerPath]):
        # TODO
        # Do we need fixed paths?
        # All empty paths are the only fixed paths now.
        fixed = [p for p in paths if not p]
        explore = [p for p in paths if p]
        return fixed, [], explore

    @classmethod
    def __block_node_search_space(self, block: BlockNode):
        explore = [LayerPath(p) for p in block.sub_paths if p]
        fixed = ([] if block.is_leaf else [LayerPath(block.main_path)]) \
            + [LayerPath(p) for p in block.sub_paths if not p]
        return [], fixed, explore

    def __get_search_space(self) -> Generator:
        fixed_head = self._fixed_head
        fixed_tail = self._fixed_tail
        explore = self._explore
        for s in permutations(explore, len(explore)):
            yield PathOrder(fixed_head + list(s) + fixed_tail)

    def __iter__(self):
        return self.__get_search_space()
