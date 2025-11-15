from __future__ import annotations

import logging
from copy import copy
from sys import maxsize
from typing import TYPE_CHECKING

from config import cfg
from logger import init_logger
from software.compiler.compile_info_analyzer import CompileInfoAnalyzer
from software.compiler.layer_compile import LayerCompile
from software.compiler.optimizer import Minimizer

if TYPE_CHECKING:
    from typing import Tuple

    from software.compiler.compile_info import CompileInfo
    from software.network import Layer, LayerBlock, LayerPath
    from software.network.block_node import BlockNode

    from .path_order_space import PathOrder

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang", "Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("Block Node Optimizer", logging.INFO)

def acc_cost(cost1, cost2):
    if cost1 == maxsize:
        return maxsize
    if cost2 == maxsize:
        return maxsize
    return cost1 + cost2

class BlockNodeOptimizer(object):
    def __init__(self, block: BlockNode, info: CompileInfo, in_stationary: int = 0):
        from .path_order_space import PathOrderSpace
        from .stationary_space import BlockStationary

        self._block = block
        self._stationary = BlockStationary(block)
        self._paths = PathOrderSpace.from_block_node(block)
        self._initial_info = info
        self._source_idx = info.num_compiled
        self._input_stationary = in_stationary
        self._min_stationary = cfg.MIDAP.FMEM.NUM - 1

    def optimize_block(self) -> CompileInfo:
        info = self._initial_info
        info.append_layer(self._block.source)
        logger.debug(f"Compile Source Layer: {self._block.source}")
        if not LayerCompile.compile(info):
            raise RuntimeError(
                f"Could not compile. FMEM size may be too small for this model."
            )
        if len(self._block.layers) > 1:
            return self.explore_order_stationary()
        else:
            return info

    def explore_order_stationary(self) -> CompileInfo:
        logger.debug("=============== [ Start  Searching Order ] ===============")
        optimizer = Minimizer(self.__explore_order, self._paths)
        optimizer.optimize()
        logger.debug("=============== [ Finish Searching Order ] ===============\n")
        return optimizer.item

    # Optimize block (path order)
    def __explore_order(self, order: PathOrder) -> Tuple[CompileInfo, int]:
        cost = 0
        info = copy(self._initial_info)

        # Explore all paths except the last path
        logger.debug(f"Searching: {order}")
        for p in filter(lambda p: p, order[:-1]):
            info, _cost = self.__explore_stationary(p, info)
            cost = acc_cost(cost, _cost)

        # Last path
        # logger.info(f"len:{len(order[-1])}")
        if len(order[-1]) > 0:
            info, _cost = self.__get_path_cost(order[-1], self._input_stationary, info)
            cost = acc_cost(cost, _cost)
            _cost = self.__compile_and_calc_cost(info)
            cost = acc_cost(cost, _cost)
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

        if optimizer.item is None:
            logger.error(f"Path [{path}] cannot be compiled.")
            raise RuntimeError
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
        if not path._path:
            return info, 0
        logger.debug(f"Searching: Stationary({in_stationary})")
        _info = copy(info)
        path.input_stationary = in_stationary
        self.blk2layer_info(path, _info)
        cost = self.__compile_and_calc_cost(_info)
        if _info.fmem_info.num_available_banks <= 0:
            cost = maxsize
        _cost = self.__calc_stationary_cost(in_stationary)
        logger.debug(f"  => calc cost: {cost}, stationary cost: {_cost}")
        cost = acc_cost(cost, _cost)
        logger.debug(f"  => Total Cost: {cost}")
        return _info, cost

    def __append_blk(self, b: LayerBlock, info: CompileInfo, _in_stationary: int):
        if not LayerCompile.compile(info):
            raise NotImplementedError
        in_statinoary = _in_stationary if self._block.source == b.source else 0
        blk_opt = BlockNodeOptimizer(b, info, in_statinoary)
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
