from __future__ import annotations

import logging
from collections import OrderedDict
from functools import reduce
from typing import TYPE_CHECKING

import attr
from logger import init_logger

if TYPE_CHECKING:
    from typing import List, Tuple

    from software.network.model import ModelGraph
    from software.system_compiler.prefetch_info import PrefetchInfo

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"

logger = init_logger("Dependency Info", logging.INFO)


@attr.s(slots=True, init=True)
class DependencyInfo(object):
    model_dict: OrderedDict[int, ModelGraph] = attr.ib(
        default=attr.Factory(OrderedDict)
    )
    layer2id: OrderedDict[int, OrderedDict[str, int]] = attr.ib(
        default=attr.Factory(OrderedDict)
    )  # int : core id, [str, int] : [layer name, layer id]
    global_dependency: OrderedDict[str, List[str]] = attr.ib(
        default=attr.Factory(OrderedDict)
    )  # str : Layer Name
    core_dependency: OrderedDict[
        int, OrderedDict[str, List[Tuple[int, str]]]
    ] = attr.ib(
        default=attr.Factory(OrderedDict)
    )  # Not in use..
    core_idx = attr.ib(default=-1)

    def register_prefetch_info(self, info: OrderedDict[List[PrefetchInfo]]):
        all: List[PrefetchInfo] = reduce(lambda acc, cur: acc + cur, info.values(), [])
        core_id = self.core_idx-1
        if core_id not in self.layer2id:
            self.layer2id[core_id] = OrderedDict()
        else:
            raise RuntimeError(
                f"register core info must be called once per each core, id {core_id}"
            )
        prefetch_id = 0
        if core_id not in self.core_dependency:
            self.core_dependency[core_id] = OrderedDict()
        for prefetch in all:
            pf_name = prefetch.name
            self.layer2id[core_id][pf_name] = prefetch_id
            prefetch_id += 1
            if pf_name not in self.core_dependency[core_id]:
                self.core_dependency[core_id][pf_name] = list()

    def register_core_info(self, core_id: int, model: ModelGraph):
        self.model_dict[core_id] = model
        if core_id not in self.layer2id:
            self.layer2id[core_id] = OrderedDict()
            if self.core_idx < 0 or self.core_idx > core_id:
                self.core_idx = core_id
        else:
            raise RuntimeError(
                f"register core info must be called once per each core, id {core_id}"
            )
        layer_id = 0
        if core_id not in self.core_dependency:
            self.core_dependency[core_id] = OrderedDict()
        for layer in model.layers:
            layer_name = layer.name
            self.layer2id[core_id][layer_name] = layer_id
            layer_id += 1
            if layer_name not in self.core_dependency[core_id]:
                self.core_dependency[core_id][layer_name] = list()

    def add_core_dependency(self, dst: Tuple[int, str], src: Tuple[int, str]):
        dst_core_id, dst_layer_name = dst
        if dst_core_id not in self.core_dependency:
            self.core_dependency[dst_core_id] = OrderedDict()
        if dst_layer_name not in self.core_dependency[dst_core_id]:
            self.core_dependency[dst_core_id][dst_layer_name] = list()
        self.core_dependency[dst_core_id][dst_layer_name].append(src)

    def add_global_dependency(self, dst: str, src: str):
        if dst in self.global_dependency:
            self.global_dependency[dst].append(src)
        else:
            self.global_dependency[dst] = [src]

    def get_core_dependency_info(
        self, core_id
    ) -> OrderedDict[str, List[Tuple[int, int]]]:
        if core_id not in self.core_dependency:
            raise RuntimeError(
                f"Core {core_id} must be registered before calling get_core_dependency_info"
            )
        else:
            ret_dict = OrderedDict()
            for layer_name in self.layer2id[core_id]:
                core_dependency_list = list(
                    map(
                        lambda x: (x[0], self.layer2id[x[0]][x[1]]),
                        self.core_dependency[core_id][layer_name],
                    )
                )  ### Warning: Core dependency dictionary is not tested
                global_dependency_list = self._get_global_dependency(
                    core_id, layer_name
                )
                ret_dict[layer_name] = core_dependency_list + global_dependency_list
            return ret_dict

    def get_core_layer_id_info(self, core_id: int) -> OrderedDict[str, Tuple[int, int]]:
        ret = OrderedDict()
        for layer_name in self.layer2id[core_id]:
            ret[layer_name] = tuple([core_id, self.layer2id[core_id][layer_name]])
        return ret

    def _get_global_dependency(self, dst_core_id, layer_name) -> List[Tuple[int, int]]:
        def _check_connection(model: ModelGraph, src: str, dst: str):
            if src not in model._str2layer:
                return False
            if dst not in model._str2layer:
                return False
            src_layer = model._str2layer[src]
            dst_layer = model._str2layer[dst]
            return dst_layer in src_layer.inputs

        if layer_name not in self.global_dependency:
            return list()
        ret = list()
        src_layers = self.global_dependency[layer_name]
        for src_layer_name in src_layers:
            if dst_core_id in self.layer2id and dst_core_id in self.model_dict:
                model = self.model_dict[dst_core_id]
                if _check_connection(model, src_layer_name, layer_name):
                    continue
            for core_id in self.layer2id:
                # if core_id == dst_core_id and dst_core_id in self.model_dict:
                #     # Check connection
                #     model = self.model_dict[dst_core_id]
                #     if model.check_connection(src_layer_name, layer_name):
                #         continue
                if src_layer_name in self.layer2id[core_id]:
                    src_layer_id = self.layer2id[core_id][src_layer_name]
                    ret.append(tuple([core_id, src_layer_id]))
        return ret
