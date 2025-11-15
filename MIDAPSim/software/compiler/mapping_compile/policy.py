from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING
from software.network.types import OpType

from logger import init_logger
from software.network.block_graph import BlockGraph
from software.network.layer import Layer
from software.network.model import ModelGraph

if TYPE_CHECKING:
    from typing import List

    from software.network.block_node import BlockNode
    from software.system_compiler.system_info import SystemInfo



__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang", "Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"

logger = init_logger("Mapping Compile", logging.INFO)


class MappingCompilePolicy(ABC):
    def __init__(self, info: SystemInfo) -> None:
        self._info = info
        self.mapping: List[ModelGraph] = []
        logger.info("Mapping Compile starts")

    @property
    def model(self):
        return self._info.shared_compile_info.model

    @property
    def dependency_info(self):
        return self._info.dependency_info

    @property
    def core_info(self):
        return self._info.core_info

    @abstractmethod
    def compile(self):
        pass


class SingleCoreMappingCompile(MappingCompilePolicy):
    def compile(self):
        from copy import copy

        self.mapping.append(copy(self.model))

class BlockGraphPipeline(SingleCoreMappingCompile):  # Pipelining
    def compile(self):
        self._preprocess()
        self._run()

    def _preprocess(self) -> None:
        self._blk_graph = BlockGraph()
        self._blk_graph.build(self.model)
        logger.info(f"BlockGraph is builded: {self._blk_graph._graph}")

    def _run(self) -> None:
        import math

        paths = []
        num_paths = len(self.core_info)
        blocks = self._blk_graph.ordered_dfs()
        blk_idx = 0
        while num_paths > 0:
            if num_paths > 1 and (len(blocks) - blk_idx) == 1:
                paths += self._divide_single_block_to_paths(
                    self._blk_graph[blocks[-1]], num_paths
                )
                num_paths = 0
            else:
                num_allocate_blocks = math.ceil((len(blocks) - blk_idx) / num_paths)
                paths.append(
                    self._block_set_to_layers(
                        blocks[blk_idx : blk_idx + num_allocate_blocks]
                    )
                )
                num_paths -= 1
                blk_idx += num_allocate_blocks
        self._check_result(paths)
        self.mapping = [
            self._layers_to_model(layers, cfg.core_id)
            for layers, cfg in zip(paths, self.core_info)
        ]

    def _check_result(self, paths: List[List[Layer]]):
        paths_set = [set(path) for path in paths]
        for p1, p2 in zip(paths, paths_set):
            if len(p1) != len(p2):
                raise RuntimeError(f"Duplicated Layer exists in path {p1}")
        logger.debug(f"Check duplication among paths")
        for p1 in paths_set:
            for p2 in paths_set:
                if p1 != p2 and len(p1 & p2) > 0:
                    raise RuntimeError(f"Duplicated Layer exists between {p1}, {p2}")
        logger.debug(f"Finish duplication check")
        len_paths = [len(path) for path in paths_set]
        if sum(len_paths) != len(self.model):
            raise RuntimeError(
                f"Wrong mapping.. mapped {sum(len_paths)} vs origin {len(self.model)}"
            )

    def _divide_single_block_to_paths(
        self, block: BlockNode, num_paths: int
    ) -> List[List[Layer]]:
        import math

        paths = block.all_paths
        if len(paths) < num_paths:
            path = reduce(lambda acc, cur: acc + cur, paths, [])
            return self._divide_single_path_to_paths(path, num_paths)
        ret_paths = []
        path_idx = 0
        while num_paths > 0:
            num_allocate_path = math.ceil((len(paths) - path_idx) / num_paths)
            ret_paths.append(
                reduce(
                    lambda acc, cur: acc + cur,
                    paths[path_idx : path_idx + num_allocate_path],
                    [],
                )
            )
            path_idx += num_allocate_path
            num_paths -= 1
        return ret_paths

    def _divide_single_path_to_paths(
        self, path: List[Layer], num_paths: int
    ) -> List[List[Layer]]:
        import math

        path_idx = 0
        ret_paths = []
        while num_paths > 0:
            num_allocate_path = math.ceil((len(path) - path_idx) / num_paths)
            ret_paths.append(path[path_idx : path_idx + num_allocate_path])
            num_paths -= 1
            path_idx += num_allocate_path
        return ret_paths

    def _block_set_to_layers(self, blocks: List[int]) -> List[Layer]:
        logger.info(f"Blocks to be mapped: {list(blocks)}")
        layers_list = [self._blk_graph[id].layers for id in blocks]
        layers = reduce(lambda acc, cur: acc + cur, layers_list, [])
        return layers

    def _layers_to_model(self, layers: List[Layer], core_id=-1) -> ModelGraph:
        from copy import copy

        model = self.model
        new_model = ModelGraph(f"{model.name}_Core_{core_id}")
        new_layer_dict = {l: copy(l) for l in layers}
        layer: Layer
        new_layer: Layer
        for layer, new_layer in new_layer_dict.items():
            for in_layer in model.get_incoming_nodes(layer):
                if in_layer in new_layer_dict:
                    new_layer.inputs.append(new_layer_dict[in_layer])
            for out_layer in model.get_next_node(layer):
                if out_layer in new_layer_dict:
                    new_layer.outputs.append(new_layer_dict[out_layer])
            logger.debug(
                f"New layer: {new_layer}, Inputs : {new_layer.inputs}, Input Vtensors: {new_layer.in_vtensors}"
            )
            new_model.register_layer(new_layer)
        logger.debug(f"Mapped model for Core {core_id}: {new_model._graph}")
        logger.debug(f"Inputs: {new_model.inputs}")
        return new_model

class LayerPipeline(BlockGraphPipeline):  # Pipelining
    def _run(self) -> None:
        num_paths = len(self.core_info)
        blocks = self._blk_graph.ordered_dfs()
        layers = self._block_set_to_layers(blocks)
        paths = self._divide_single_path_to_paths(
            layers, num_paths
        )
        self._check_result(paths)
        self.mapping = [
            self._layers_to_model(layers, cfg.core_id)
            for layers, cfg in zip(paths, self.core_info)
        ]

    def _divide_single_path_to_paths(
        self, path: List[Layer], num_paths: int
    ) -> List[List[Layer]]:
        overhead_arr = [l.estimate_overhead() for l in path]
        path_idx = 0
        ret_paths = []
        while num_paths > 0:
            total_overhead = sum(overhead_arr[path_idx:])
            next_idx = path_idx + 1
            while next_idx < len(path):
                if (total_overhead / num_paths) < sum(overhead_arr[path_idx:next_idx]):
                    break
                next_idx += 1
            if num_paths > 1:
                while path[next_idx-1].op.type == OpType.Dummy: # Temporal solution: leave dummy layers for the next core
                    next_idx -= 1
            ret_paths.append(path[path_idx : next_idx])
            num_paths -= 1
            path_idx = next_idx
        return ret_paths

class HostMappingCompile(BlockGraphPipeline):
    def _run(self) -> None:
        num_paths = len(self.core_info)
        blocks = self._blk_graph.ordered_dfs()
        layers = self._block_set_to_layers(blocks)
        paths = self._divide_single_path_to_paths(
            layers, num_paths
        )
        self._check_result(paths)
        self.mapping = [
            self._layers_to_model(layers, cfg.core_id)
            for layers, cfg in zip(paths[:-1], self.core_info[:-1])
        ] + [self._layers_to_host_model(paths[-1], self.core_info[-1].core_id)]

    def _divide_single_path_to_paths(
        self, path: List[Layer], num_paths: int
    ) -> List[List[Layer]]:
        host_path = list(filter(lambda x : x.op.type == OpType.HostProcess, path))
        path = list(filter(lambda x : x.op.type != OpType.HostProcess, path))
        path_idx = 0
        ret_paths = []
        num_paths = num_paths - 1
        overhead_arr = [l.estimate_overhead() for l in path]
        while num_paths > 0:
            total_overhead = sum(overhead_arr[path_idx:])
            next_idx = path_idx + 1
            while next_idx < len(path):
                if (total_overhead / num_paths) < sum(overhead_arr[path_idx:next_idx]):
                    break
                next_idx += 1
            ret_paths.append(path[path_idx : next_idx])
            num_paths -= 1
            path_idx = next_idx
        ret_paths.append(host_path)
        return ret_paths
    
    def _layers_to_host_model(self, layers: List[Layer], core_id=-1) -> ModelGraph:
        from copy import copy
        model = self.model
        new_model = ModelGraph(f"{model.name}_Core_{core_id}")
        new_layer_dict = {l: copy(l) for l in layers}
        layer: Layer
        new_layer: Layer
        for layer, new_layer in new_layer_dict.items():
            logger.debug(
                f"New layer: {new_layer}, Inputs : {new_layer.inputs}, Input Vtensors: {new_layer.in_vtensors}"
            )
            new_model.register_layer(new_layer)
        logger.debug(f"Mapped model for Core {core_id}: {new_model._graph}")
        logger.debug(f"Inputs: {new_model.inputs}")
        return new_model

class MultiCoreBBGraPCompile(BlockGraphPipeline):

    def _run(self) -> None:
        try:
            from externals.bbgrap.package import BBGraP
        except ImportError as e:
            raise ImportError("Couldn't find BBGraP external module")

        bbgrap = BBGraP()
        info_basic = bbgrap.load_graph_basic_info('test_store', "externals/bbgrap/package/config_files/")
        map_dict = {}

        for layer in self.model.layers:
            layer_info = info_basic[layer.name]
            layer_core = layer_info['core']

            if layer_core == -1:
                layer_core = 0

            if layer_core not in map_dict:
                map_dict[layer_core] = []
            map_dict[layer_core].append(layer)


        self.mapping = [
            self._layers_to_model(layers, cfg.core_id)
            for layers, cfg in zip(list(map_dict.values()), self.core_info)
        ]

        return
