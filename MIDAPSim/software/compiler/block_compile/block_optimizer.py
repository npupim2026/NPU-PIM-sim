from __future__ import annotations

import logging
from copy import copy
from sys import maxsize
from typing import TYPE_CHECKING

from config import cfg
from logger import init_logger
from software.compiler.compile_info_analyzer import CompileInfoAnalyzer
from software.compiler.layer_compile.compile import LayerCompile
from software.compiler.optimizer import Minimizer

if TYPE_CHECKING:
    from typing import Tuple

    from software.compiler.compile_info import CompileInfo
    from software.network import Layer, LayerBlock, LayerPath

    from .path_order_space import PathOrder

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


logger = init_logger("Block Optimizer", logging.DEBUG)


class BlockOptimizer(object):
    def __init__(self, block: LayerBlock, info: CompileInfo, in_stationary: int = 0):
        from .path_order_space import PathOrderSpace
        from .stationary_space import BlockStationary

        self._block = block
        self._stationary = BlockStationary(block)
        self._order_space = PathOrderSpace.from_layer_block(self._block)
        self._initial_info = info
        self._source_idx = info.num_compiled
        self._input_stationary = in_stationary
        self._min_stationary = cfg.MIDAP.FMEM.NUM - 1

    def explore_order_stationary(self) -> CompileInfo:
        logger.debug("=============== [ Start  Searching Order ] ===============")
        optimizer = Minimizer(self.__explore_order, self._order_space)
        optimizer.optimize()
        logger.debug("=============== [ Finish Searching Order ] ===============\n")
        return optimizer.item

    # Optimize block (path order)
    def __explore_order(self, order: PathOrder) -> Tuple[CompileInfo, int]:
        cost = 0
        info = copy(self._initial_info)
        sink = self._block.sink

        # Explore all paths except the last path
        sink.set_prev(order.get_last_layer())
        logger.debug(f"Searching: {order} {info.fmem_info}")
        for p in filter(lambda p: p, order[:-1]):
            info, _cost = self.__explore_stationary(p, info)
            cost += _cost

        # Last path
        info, _cost = self.__get_path_cost(order[-1], self._input_stationary, info)
        cost += _cost

        info.append_layer(sink)
        cost += self.__compile_and_calc_cost(info)
        return info, cost

    def __compile_and_calc_cost(self, info: CompileInfo) -> int:
        return (
            CompileInfoAnalyzer.off_chip_access(info, self._source_idx)
            if LayerCompile.compile(info)
            else maxsize
        )

    # Optimizer path (stationary)
    def __explore_stationary(
        self, path: LayerPath, info: CompileInfo
    ) -> Tuple[CompileInfo, int]:
        def explore_stationary(s: int):
            return self.__get_path_cost(path, s, info)

        logger.debug("============== [ Start  Searching Stationary ] ==============")
        optimizer = Minimizer(explore_stationary, self._stationary.get_input_range())
        optimizer.optimize()

        curr_info: CompileInfo = optimizer.item
        curr_stationary = curr_info.get_layer_info(path[0]).stationary.input
        self._min_stationary = min(curr_stationary, self._min_stationary)
        logger.debug("============== [ Finish Searching Stationary ] ==============")
        return optimizer.item, optimizer.cost

    def __calc_stationary_cost(self, in_stationary: int) -> int:
        t = self._block.source.out_vtensor
        prev_save_size = min(self._min_stationary * t.size_per_bank, t.total_size)
        curr_save_size = min(in_stationary * t.size_per_bank, t.total_size)
        # logger.debug(f"Prev: {prev_save_size} Curr: {curr_save_size}")
        return max(0, prev_save_size - curr_save_size)

    def __get_path_cost(
        self, path: LayerPath, in_stationary: int, info: CompileInfo
    ) -> Tuple[CompileInfo, int]:
        logger.debug(f"Searching: Stationary({in_stationary})")
        _info = copy(info)
        path.input_stationary = in_stationary
        self.blk2layer_info(path, _info)
        cost = self.__compile_and_calc_cost(_info)
        if cost == maxsize:
            pass
        else:
            cost += self.__calc_stationary_cost(in_stationary)
        logger.debug(f"  => Cost: {cost}")
        return _info, cost

    def __append_blk(self, b: LayerBlock, info: CompileInfo, _in_stationary: int):
        if not LayerCompile.compile(info):
            raise NotImplementedError
        in_statinoary = _in_stationary if self._block.source == b.source else 0
        blk_opt = BlockOptimizer(b, info, in_statinoary)
        _info: CompileInfo = blk_opt.explore_order_stationary()
        info.update_layer_dict(_info.layer_dict)

    def __append_layer(self, l: Layer, info: CompileInfo, in_stationary: int):
        info.append_layer(l)
        if self._block.source in l.inputs:
            info.get_layer_info(l).stationary.input = in_stationary

    def blk2layer_info(self, path: LayerPath, info: CompileInfo):
        def layer_func(l: Layer):
            self.__append_layer(l, info, path.input_stationary)

        def block_func(b: LayerBlock):
            self.__append_blk(b, info, path.input_stationary)

        path.traverse(layer_func=layer_func, block_func=block_func, ordered=True)
