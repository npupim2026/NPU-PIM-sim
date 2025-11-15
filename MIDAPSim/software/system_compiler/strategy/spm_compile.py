from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from software.network.model import ModelGraph
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Set, Tuple

import attr
from logger import init_logger
from software.network.layer import Layer
from software.network.tensor import Tensor
from software.network.types import TensorType
from software.system_compiler.memory_info import MemoryType
from software.system_compiler.prefetch_info import PrefetchInfo

from .strategy import Strategy

if TYPE_CHECKING:
    from software.system_compiler.system_info import SystemInfo


__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("SPM Compile", logging.DEBUG)


class SPMCompile(Strategy):
    def _print_status(self, curr):
        def _print_func(l):
            return [str(x) for x in l]

        logger.debug("====================================================")
        logger.debug("====================================================")

    def _initialize(self, info: SystemInfo) -> None:
        self._shared_info = info.shared_compile_info
        self._model = info.shared_compile_info.model
        self._dep_info = info.dependency_info
        self._core_info = info.core_info
        self._memory_info = info.memory_info

    def _preprocess(self) -> None:
        # Define shared items
        pass

    def compile(self, info: SystemInfo) -> SystemInfo:
        if not info.memory_info.spm_info.in_use:
            return info
        self._initialize(info)
        self._preprocess()
        self._run()
        self._postprocess()
        info.shared_compile_info.model = self._model
        info.dependency_info = self._dep_info
        info.memory_info = self._memory_info
        return info

    def _run(self) -> None:
        pass

    def _postprocess(self) -> None:
        pass


@attr.s(init=True, slots=True, repr = False)
class BufInfo(object):
    tensor: Tensor = attr.ib()
    head: int = attr.ib()
    size: int = attr.ib()

    @property
    def tail(self):
        return self.head + self.size

    def __repr__(self):
        return f"({self.tensor.name}: {self.head}, {self.size})"


class Prototype(SPMCompile):
    ## 1. Assume only 2 banks, one for temporal data & shared data (bank 1) and one for constant (bank 0)
    def __init__(self):
        self.logger = init_logger("SPM Compile2", logging.DEBUG)
    def _preprocess(self) -> None:
        minfo = self._memory_info
        if minfo.spm_info.num_banks != 2:
            raise RuntimeError(
                "Cannot compile SPM.. Prototype requires only 2 banks for SPM"
            )
        self.__check_feature_space_available()
        str2tensor = self._shared_info.model._str2tensor
        tensors = [str2tensor[name] for name in minfo.shared_address_dict]
        self._shared_features = list(
            filter(lambda x: x.type != TensorType.Constant, tensors)
        )
        self._shared_constants = list(
            filter(lambda x: x.type == TensorType.Constant, tensors)
        )
        self._tensor_dep_dict = self.__make_tensor_dependency_dict()

    def _run(self):
        feature_bank_prefetch_info = self.__setup_feature_spm_info()
        constants_bank_prefetch_info = self.__setup_constant_spm_info()
        self._memory_info.spm_info.bank_prefetch_info = {
            0: constants_bank_prefetch_info,
            1: feature_bank_prefetch_info,
        }

    def _postprocess(self) -> None:
        self._dep_info.register_prefetch_info(
            self._memory_info.spm_info.bank_prefetch_info
        )

    def __check_feature_space_available(self):
        minfo = self._memory_info
        # Check available buffer size
        required_bank_size = minfo.data_shared_feature.size
        for _, lmi in minfo.local_memory_info_dict.items():
            required_bank_size += lmi.required_size
        if required_bank_size > minfo.spm_info.bank_size:
            raise RuntimeError(
                f"required buffer size is too big {required_bank_size} vs {minfo.spm_info.bank_size}"
            )

    def __make_tensor_dependency_dict(self):
        model = self._shared_info.model

        def _add_tensor_dependency(
            layer: Layer, dep_dict: DefaultDict[Tensor, Set[Layer]]
        ):
            tensors = layer.get_tensors()
            if tensors:
                dep_dict[tensors[0]].add(layer)
            if layer.in_tensor is not None and layer.in_tensor.type == TensorType.In:
                dep_dict[layer.in_tensor].add(layer)

        dep_dict: DefaultDict[Tensor, Set[Layer]] = defaultdict(set)
        func = lambda l: _add_tensor_dependency(l, dep_dict)
        model.traverse(func, topological=True)
        return dep_dict

    def __setup_feature_spm_info(self):
        minfo = self._memory_info
        addr_offset = minfo.spm_info.bank_size  # assume bank 1
        bank_prefetch_info = []
        for tensor in self._shared_features:
            if tensor.type == TensorType.In:
                prefetch_id = f"Prefetch_{tensor.name}"
                offchip_id = f"Offchip_{tensor.name}"
                pfinfo = PrefetchInfo(
                    prefetch_id, offchip_id, tensor.name, tensor.data.size
                )
                for layer in self._tensor_dep_dict[tensor]:
                    self._dep_info.add_global_dependency(layer.name, prefetch_id)
                bank_prefetch_info.append(pfinfo)
                old_addr = minfo.shared_address_dict[tensor.name]
                minfo.shared_address_dict[offchip_id] = old_addr
            minfo.shared_address_dict[tensor.name] = [MemoryType.SPM.value, addr_offset]
            addr_offset += tensor.data.size
        for _, lmi in minfo.local_memory_info_dict.items():
            new_local_addr_dict = OrderedDict()
            for name, addr in lmi.address_dict.items():
                new_local_addr_dict[name] = addr + addr_offset
            lmi.address_dict = new_local_addr_dict
            addr_offset += lmi.required_size
        return bank_prefetch_info

    def __setup_constant_spm_info(self):
        def _tsize(tensors: List[Tensor]) -> int:
            return sum([t.data.size for t in tensors])

        minfo = self._memory_info
        bsize = minfo.spm_info.bank_size
        model = self._model
        bank_prefetch_info = []
        processed_layers: Set[Layer] = set()
        prefetched_layers: List[Tuple[int, Layer]] = list()  # core id, layer
        processed_tensors: Set[Tensor] = set()
        core_candidates: List[List[str]] = [
            [l.name for l in c.compile_info.layer_dict.keys()] for c in self._core_info
        ]
        # core_candidates = [
        #     list(filter(lambda l: len(model._str2layer[l].op.get_tensors()) > 0, cand))
        #     for cand in core_candidates
        # ]
        buf_info: List[BufInfo] = list()
        layer_candidates = list(filter(lambda l: not model.get_incoming_nodes(l), model.inputs))
        # layer_candidates: List[Layer] = Prototype.__update_layer_candidates(
        #     model, layer_candidates, None
        # )
        runnable: List[Tuple[int, Layer]] = []  # core id, layer

        def _select_target(runnable: List[Tuple[int, Layer]]) -> Tuple[int, Layer]:
            from sys import maxsize
            constant_size = maxsize
            for layer in runnable:
                tensors = layer[1].get_tensors()
                if not tensors:
                    return layer
                if set(tensors).issubset(processed_tensors):
                    return layer
                ccs = _tsize(tensors)
                if ccs < constant_size:
                    select = layer
                    constant_size = ccs
            return select

        def _check_available(binfo, size):
            proc_all = processed_layers | set([x[1] for x in prefetched_layers])
            check_binfo = self.filtering_buf_info(binfo, proc_all)
            return Prototype.find_allocate_location(check_binfo, bsize, size) >= 0

        while len(layer_candidates) > 0 or len(runnable) > 0:
            if not runnable:
                runnable = Prototype.update_core_candidates(
                    model, processed_layers | set([x[1] for x in prefetched_layers]), layer_candidates, core_candidates
                )
                if not runnable:
                    raise RuntimeError("Model Error : No runnable layers left")
            logger.debug("================================================")
            logger.debug(f"Updated Layer Candidates: {layer_candidates}")
            logger.debug(f"Updated Core Candidates: {core_candidates}")
            logger.debug(f"Updated Runnable: {runnable}")
            selected = _select_target(runnable)
            idx, target = selected
            logger.debug(f"Selected: {target}")
            runnable.remove(selected)
            tensors = target.get_tensors()
            if not tensors:
                prefetched_layers.append((idx, target))
                layer_candidates = Prototype.update_layer_candidates(
                    model, layer_candidates, target
                )
                continue
                # raise RuntimeError("Layers w/o constants must be pruned")
            required_size = _tsize(tensors)
            logger.debug(
                f"Check Available to prefetch: {tensors[0].name}, size {required_size}"
            )
            if _check_available(buf_info, required_size) and not set(tensors).issubset(
                processed_tensors
            ):
                logger.debug(f"Available: current buf_info = {buf_info}")
                pruned_layers: List[Tuple[int, Layer]] = []
                while True:
                    head = Prototype.find_allocate_location(
                        buf_info, bsize, required_size
                    )
                    if head >= 0:
                        break
                    if not prefetched_layers:
                        raise RuntimeError(
                            "Unexpected situation(infinite loop): this case should be pruned"
                        )
                    proc = prefetched_layers[0]
                    processed_layers.add(proc[1])
                    pruned_layers = list(
                        filter(lambda x: x[0] != proc[0], pruned_layers)
                    ) + [proc]
                    prefetched_layers = prefetched_layers[1:]
                    buf_info = self.filtering_buf_info(buf_info, processed_layers)
                    logger.debug(f"==================================")
                    logger.debug(f"Fail to allocate -> process {proc}")
                    logger.debug(f"Pruned layers: {pruned_layers}")
                    logger.debug(f"BufInfo -> {buf_info}")
                insert = BufInfo(tensors[0], head, required_size)
                buf_info = Prototype.allocate_buf_info(buf_info, insert)
                prefetch_id = f"Prefetch_{tensors[0].name}"
                offchip_id = f"Offchip_{tensors[0].name}"
                pfinfo = PrefetchInfo(
                    prefetch_id, offchip_id, tensors[0].name, required_size
                )
                for _, layer in pruned_layers:
                    self._dep_info.add_global_dependency(prefetch_id, layer.name)
                for layer in self._tensor_dep_dict[tensors[0]]:
                    self._dep_info.add_global_dependency(layer.name, prefetch_id)
                bank_prefetch_info.append(pfinfo)
                minfo.shared_address_dict[offchip_id] = minfo.shared_address_dict[
                    tensors[0].name
                ]
                addr = head
                for t in tensors:
                    minfo.shared_address_dict[t.name] = [MemoryType.SPM.value, addr]
                    addr += t.data.size
            prefetched_layers.append((idx, target))
            processed_tensors = processed_tensors.union(set(tensors))
            layer_candidates = Prototype.update_layer_candidates(
                model, layer_candidates, target
            )
        return bank_prefetch_info

    @staticmethod
    def update_layer_candidates(
        model : ModelGraph,
        # processed_layers : Set[Layer],
        candidates: Set[Layer],
        processed: Layer = None
        ):
        to_process = set()
        remain = set()
        if not candidates:
            return set()
        for layer in candidates:
            if layer == processed:
                to_process = to_process | model.get_next_node(layer)
            # elif not layer.op.get_tensors():
            #     to_process = to_process | model.get_next_node(layer)
            #     processed_layers.add(layer)
            else:
                remain.add(layer)
        return remain | Prototype.update_layer_candidates(model, to_process, None)

    @staticmethod
    def update_core_candidates(
        model : ModelGraph, processed_layers : Set[Layer], layer_candidates: Set[Layer], core_candidates: List[List[str]]
    ) -> List[Tuple[int, Layer]]:
        str_candidates = {l.name: l for l in layer_candidates}
        runnable_idx = []
        for idx in range(len(core_candidates)):
            candidate = core_candidates[idx]
            if not candidate:
                continue
            if candidate[0] in str_candidates:
                runnable_idx.append(idx)
        runnable = []
        for idx in runnable_idx:
            candidates = core_candidates[idx]
            candidate = str_candidates[candidates[0]]
            if set(model.get_incoming_nodes(candidate)).issubset(processed_layers):
                runnable.append((idx, candidate))
                core_candidates[idx] = candidates[1:]
        return runnable

    @staticmethod
    def find_allocate_location(
        buf_info: List[BufInfo], buf_size: int, size: int
    ) -> int:
        head = 0
        for info in buf_info:
            if info.head - head >= size:
                return head
            head = info.tail
        if buf_size - head >= size:
            return head
        return -1

    @staticmethod
    def allocate_buf_info(buf_info: List[BufInfo], insert: BufInfo) -> List[BufInfo]:
        idx = 0
        for info in buf_info:
            if insert.tail < info.head:
                break
            idx += 1
        return buf_info[:idx] + [insert] + buf_info[idx:]

    def filtering_buf_info(self, buf_info: List[BufInfo], layer_set: Set[Layer]):
        def _filter_func(b: BufInfo):
            return set(self._tensor_dep_dict[b.tensor]).issubset(layer_set)

        remained = filter(lambda x: not _filter_func(x), buf_info)
        return list(remained)


class N2NStrategy(Prototype):
    ## 1. Assume only 2 banks, one for temporal data & shared data (bank 1) and one for constant (bank 0)
    def __init__(self):
        self.logger = init_logger("SPM Compile: N2N", logging.DEBUG)
    def _preprocess(self) -> None:
        minfo = self._memory_info
        if minfo.spm_info.num_banks != len(self._core_info):
            raise RuntimeError(
                "Cannot compile SPM.. # of MIDAP cores must be equal to # of SPM banks @ N2NStrategy"
            )
        str2tensor = self._shared_info.model._str2tensor
        tensors = [str2tensor[name] for name in minfo.shared_address_dict]
        self._shared_features = list(
            filter(lambda x: x.type not in [TensorType.Constant, TensorType.Out], tensors)
        )
        self._shared_constants = list(
            filter(lambda x: x.type == TensorType.Constant, tensors)
        )
        self._tensor_dep_dict = self.__make_tensor_dependency_dict()
        self._available_space_dict = {i: minfo.spm_info.bank_size for i in range(minfo.spm_info.num_banks)}
        self._memory_info.spm_info.bank_prefetch_info = {
            i: list() for i in range(minfo.spm_info.num_banks)
        }

    def _run(self):
        self.__setup_feature_spm_info()
        self.__setup_constant_spm_info()

    def _postprocess(self) -> None:
        self._dep_info.register_prefetch_info(
            self._memory_info.spm_info.bank_prefetch_info
        )

    def __make_tensor_dependency_dict(self):
        model = self._shared_info.model
        def _add_tensor_dependency(
            layer: Layer, dep_dict: DefaultDict[Tensor, Set[Layer]]
        ):
            tensors = layer.get_tensors()
            if tensors:
                dep_dict[tensors[0]].add(layer)
            if layer.in_tensor is not None and layer.in_tensor.type == TensorType.In:
                dep_dict[layer.in_tensor].add(layer)

        dep_dict: DefaultDict[Tensor, Set[Layer]] = defaultdict(set)
        func = lambda l: _add_tensor_dependency(l, dep_dict)
        model.traverse(func, topological=True)
        return dep_dict

    def __setup_feature_spm_info(self):
        minfo = self._memory_info
        addr_offset = minfo.spm_info.bank_size  # assume bank 1
        input_tensor_dict = {}
        tensor_dict = {}
        # Temporal data --> Corresponding SPM
        for idx, info in enumerate(self._core_info):
            model = info.compile_info.model
            tensor_dict[idx] = model.tensors
            in_tensors = []
            for l in model.inputs:
                in_tensors += [vt.tensor for vt in l.in_vtensors]
            input_tensor_dict[idx] = in_tensors
            lmi = minfo.local_memory_info_dict[info.core_id]
            addr_offset = self._available_space_dict[idx] - lmi.required_size
            if addr_offset < 0:
                raise RuntimeError(f"SPM Bank size is not sufficient for core {info.core_id}, temporal feature map data cannot be mapped to SPM")
            self._available_space_dict[idx] = addr_offset
            new_local_addr_dict = OrderedDict()
            for name, addr in lmi.address_dict.items():
                new_local_addr_dict[name] = addr + addr_offset
            lmi.address_dict = new_local_addr_dict

        # Shared features --> Corresponding SPM (Prefer the case where the shared tensor is used as input)
        for tensor in self._shared_features:
            mapped_idx = -1
            for idx in range(len(self._core_info)):
                if tensor in input_tensor_dict[idx]:
                    mapped_idx = idx
                elif tensor in tensor_dict[idx] and mapped_idx == -1:
                    mapped_idx = idx
            if mapped_idx == -1:
                raise RuntimeError(f"Cannot find mapped core idx of tensor {tensor}")
            addr_offset = self._available_space_dict[mapped_idx] - tensor.data.size
            if tensor.type == TensorType.In and addr_offset >= 0:
                prefetch_id = f"Prefetch_{tensor.name}"
                offchip_id = f"Offchip_{tensor.name}"
                pfinfo = PrefetchInfo(
                    prefetch_id, offchip_id, tensor.name, tensor.data.size
                )
                for layer in self._tensor_dep_dict[tensor]:
                    self._dep_info.add_global_dependency(layer.name, prefetch_id)
                minfo.spm_info.bank_prefetch_info[mapped_idx].append(pfinfo)
                old_addr = minfo.shared_address_dict[tensor.name]
                minfo.shared_address_dict[offchip_id] = old_addr
            if addr_offset >= 0:
                logger.debug(f"Current available space dict: {self._available_space_dict}")
                logger.debug(f"Shared Tensor {tensor} is mapped to {mapped_idx, addr_offset}")
                minfo.shared_address_dict[tensor.name] = [MemoryType.SPM.value, addr_offset + mapped_idx * minfo.spm_info.bank_size]
                self._available_space_dict[mapped_idx] = addr_offset
            elif tensor.type != TensorType.In:
                raise RuntimeError(f"SPM Bank size is not sufficient for core {mapped_idx}, shared tensor {tensor} cannot be mapped..")

    def __setup_constant_spm_info(self):
        def _tsize(tensors: List[Tensor]) -> int:
            return sum([t.data.size for t in tensors])

        minfo = self._memory_info
        model = self._model
        tensor_dict = {}
        for idx, info in enumerate(self._core_info):
            core_model = info.compile_info.model
            tensor_dict[idx] = core_model.tensors
        processed_layers: Set[Layer] = set()
        prefetched_layers: List[Tuple[int, Layer]] = list()  # core id, layer
        processed_tensors: Set[Tensor] = set()
        core_candidates: List[List[str]] = [
            [l.name for l in c.compile_info.layer_dict.keys()] for c in self._core_info
        ]
        # core_candidates = [
        #     list(filter(lambda l: len(model._str2layer[l].op.get_tensors()) > 0, cand))
        #     for cand in core_candidates
        # ]
        buf_info: Dict[int: List[BufInfo]] = {i: list() for i in range(minfo.spm_info.num_banks)}
        layer_candidates = list(filter(lambda l: not model.get_incoming_nodes(l), model.inputs))
        # layer_candidates: List[Layer] = Prototype.__update_layer_candidates(
        #     model, layer_candidates, None
        # )
        logger.debug(f"Layer Candidate: {layer_candidates}")
        runnable: List[Tuple[int, Layer]] = []  # core id, layer

        def _select_target(runnable: List[Tuple[int, Layer]]) -> Tuple[int, Layer]:
            from sys import maxsize
            constant_size = maxsize
            for layer in runnable:
                tensors = layer[1].get_tensors()
                if not tensors:
                    return layer
                if set(tensors).issubset(processed_tensors):
                    return layer
                ccs = _tsize(tensors)
                if ccs < constant_size:
                    select = layer
                    constant_size = ccs
            return select

        def _check_available(binfo, idx, size):
            proc_all = processed_layers | set([x[1] for x in prefetched_layers])
            check_binfo = self.filtering_buf_info(binfo[idx], proc_all)
            return Prototype.find_allocate_location(check_binfo, self._available_space_dict[idx], size) >= 0

        def _get_mapped_idx(mapped_tensors):
            tensor = mapped_tensors[0]
            for idx, tensors in tensor_dict.items():
                if tensor in tensors:
                    return idx
            raise RuntimeError(f"{tensor} is not mapped anywhere!")

        while len(layer_candidates) > 0 or len(runnable) > 0:
            if not runnable:
                runnable = Prototype.update_core_candidates(
                    model, processed_layers | set([x[1] for x in prefetched_layers]), layer_candidates, core_candidates
                )
                if not runnable:
                    raise RuntimeError("Model Error : No runnable layers left")
            logger.debug("================================================")
            logger.debug(f"Updated Layer Candidates: {layer_candidates}")
            logger.debug(f"Updated Core Candidates: {core_candidates}")
            logger.debug(f"Updated Runnable: {runnable}")
            selected = _select_target(runnable)
            idx, target = selected
            logger.debug(f"Selected: {target}")
            runnable.remove(selected)
            tensors = target.get_tensors()
            if not tensors:
                prefetched_layers.append((idx, target))
                layer_candidates = Prototype.update_layer_candidates(
                    model, layer_candidates, target
                )
                continue
                # raise RuntimeError("Layers w/o constants must be pruned")
            mapped_idx = _get_mapped_idx(tensors)
            required_size = _tsize(tensors)
            logger.debug(
                f"Check Available to prefetch: {tensors[0].name}, size {required_size}, mapped idx: {mapped_idx}"
            )
            if _check_available(buf_info, mapped_idx, required_size) and not set(tensors).issubset(
                processed_tensors
            ):
                logger.debug(f"Available: current buf_info = idx {mapped_idx} of {buf_info}")
                pruned_layers: List[Tuple[int, Layer]] = []
                while True:
                    head = Prototype.find_allocate_location(
                        buf_info[mapped_idx], self._available_space_dict[mapped_idx], required_size
                    )
                    if head >= 0:
                        break
                    if not prefetched_layers:
                        raise RuntimeError(
                            "Unexpected situation(infinite loop): this case should be pruned"
                        )
                    proc = prefetched_layers[0]
                    processed_layers.add(proc[1])
                    pruned_layers = list(
                        filter(lambda x: x[0] != proc[0], pruned_layers)
                    ) + [proc]
                    prefetched_layers = prefetched_layers[1:]
                    for idx in buf_info:
                        buf_info[idx] = self.filtering_buf_info(buf_info[idx], processed_layers)
                    logger.debug(f"==================================")
                    logger.debug(f"Fail to allocate -> process {proc}")
                    logger.debug(f"Pruned layers: {pruned_layers}")
                    logger.debug(f"BufInfo -> {buf_info}")
                insert = BufInfo(tensors[0], head, required_size)
                buf_info[mapped_idx] = Prototype.allocate_buf_info(buf_info[mapped_idx], insert)
                prefetch_id = f"Prefetch_{tensors[0].name}"
                offchip_id = f"Offchip_{tensors[0].name}"
                pfinfo = PrefetchInfo(
                    prefetch_id, offchip_id, tensors[0].name, required_size
                )
                for _, layer in pruned_layers:
                    self._dep_info.add_global_dependency(prefetch_id, layer.name)
                for layer in self._tensor_dep_dict[tensors[0]]:
                    self._dep_info.add_global_dependency(layer.name, prefetch_id)
                minfo.spm_info.bank_prefetch_info[mapped_idx].append(pfinfo)
                minfo.shared_address_dict[offchip_id] = minfo.shared_address_dict[
                    tensors[0].name
                ]
                addr = head + mapped_idx * minfo.spm_info.bank_size
                for t in tensors:
                    minfo.shared_address_dict[t.name] = [MemoryType.SPM.value, addr]
                    addr += t.data.size
            else:
                logger.debug("Cannot be prefetched... it should be directly fetched from DRAM.")
            prefetched_layers.append((idx, target))
            processed_tensors = processed_tensors.union(set(tensors))
            layer_candidates = Prototype.update_layer_candidates(
                model, layer_candidates, target
            )
