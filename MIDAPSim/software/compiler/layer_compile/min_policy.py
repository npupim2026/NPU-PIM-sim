from __future__ import annotations

import logging
from copy import copy
from sys import maxsize
from typing import TYPE_CHECKING

from box import Box
from logger import init_logger
from software.network.types import OpType

from .behavior import Action, Behavior
from .data_manager import DataManager
from .fmem_info import FMEMInfo
from .mapping import MappingInfo
from .policy import LayerCompliePolicy

if TYPE_CHECKING:
    from typing import Dict, List, Tuple

    from .intermediate_info import IntermediateInfo, Status
    from .layer_info import LayerInfo

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"

logger = init_logger("Min Policy", logging.INFO)


class MinPolicy(LayerCompliePolicy):
    def __init__(self, layer: LayerInfo, fmem: FMEMInfo, num_banks: Dict[str, int]):
        self.__layer_info = layer
        self.__data_manager = DataManager(layer, fmem.bank_size)
        self.__sub_data_manager : List[DataManager] = []
        if layer.op.type in [OpType.WeightedSum]:
            for i in range(1, len(layer.ordered_input_vtensors)):
                self.__sub_data_manager.append(DataManager(layer, fmem.bank_size, i))
        self.__num_bank2process = (
            fmem.num_banks if layer.is_weight_larger_than_wmem() else 1
        )
        self.__input_stationary = num_banks["input"]
        self.__num_outbanks = num_banks["output"]

        self.__behavior_stack: List[Behavior] = []
        self.best = Box({"total_size": maxsize})
        self.fail_dict = []

    @property
    def num_outbanks(self):
        return self.__num_outbanks

    @property
    def input_stationary(self):
        return self.__input_stationary

    @property
    def fmem(self):
        return self.__data_manager.fmem

    @property
    def actions(self):
        return self.__behavior_stack

    def __get_available_ops(self, force_process: bool):
        fdata2process = self.__data_manager.get_available2process(
            self.__num_bank2process
        )
        sub_proc = [sdm.get_available2process(self.__num_bank2process) for sdm in self.__sub_data_manager]
        process_ops: List[Behavior] = []
        num_process = len(fdata2process)
        if len(sub_proc) > 0:
            num_process = min(num_process, *[len(np) for np in sub_proc])
        # if not self.__layer_info.is_weight_larger_than_wmem():
        #     num_process = min(num_process, 1)
        if num_process > 0:
            data2process = fdata2process
            for sp in sub_proc:
                data2process += sp[:num_process]
            # process_ops.append(Behavior(Action.PROCESS, fdata2process[:idx]))
            process_ops.append(Behavior(Action.PROCESS, data2process))
        if force_process:
            return [], process_ops

        fdata2load = self.__data_manager.get_available2load()
        for sdm, proc in zip(self.__sub_data_manager, sub_proc):
            if len(proc) == 0:
                fdata2load = sdm.get_available2load()
        load_ops: List[Behavior] = []
        # for idx in reversed(range(1, len(fdata2load) + 1)):
        #     load_ops.append(Behavior(Action.LOAD, fdata2load[:idx]))
        if len(fdata2load) > 0:
            load_ops.append(Behavior(Action.LOAD, fdata2load[:1]))
        return load_ops, process_ops

    def __get_output_banks(self, op: Behavior):
        data_manager = self.__data_manager
        output = []
        if op.action == Action.PROCESS:
            output = data_manager.get_outbank_from_input(op.data, self.__num_outbanks)
        return output

    def search(self, info: IntermediateInfo, force_process: bool = False):
        status = info.status
        self.__data_manager.set_status(status)
        for sdm in self.__sub_data_manager:
            sdm.set_status(status)
        input_frags = self.__data_manager.get_input_data(0, self.input_stationary)
        status.fmem.set_stationary(input_frags if self.input_stationary > 0 else [])

        last_lop = None
        last_pop = None
        for op in reversed(self.__behavior_stack):
            if last_lop and last_pop:
                break
            if op.action == Action.LOAD and not last_lop:
                last_lop = op
            elif last_pop is None:
                last_pop = op
        lx = 0 if not last_lop else last_lop.data[-1].last_x
        px = 0 if not last_pop else last_pop.data[-1].last_x
        if (lx, px, self.__behavior_stack and self.__behavior_stack[-1].action == Action.PROCESS) in self.fail_dict:
            # print("*************************FAILED(DICT)****************************")
            # print("Status: {}".format([("L" if b.action == Action.LOAD else "P") + f" {b.data[-1].tensor.name} {b.data[-1].last_x}" for b in self.__behavior_stack]))
            # print(f"FMEM: {info.status.fmem}")
            # print("****************************************************************")
            return False
        load_ops, process_ops = self.__get_available_ops(force_process)
        if load_ops:
            op = load_ops[0]
            if last_lop:
                if op.data[-1].last_x == last_lop.data[-1].last_x and op.data[-1].tensor == last_lop.data[-1].tensor:
                    load_ops = []
        ops = load_ops + process_ops
        # print("Status: {}".format([("L" if b.action == Action.LOAD else "P") + f" {b.data[-1].tensor.name} {b.data[-1].last_x}" for b in self.__behavior_stack]))
        # print(f"FMEM: {info.status.fmem}")
        # print("Candidates: {}".format([("L" if b.action == Action.LOAD else "P") + f" {b.data[-1].tensor.name} {b.data[-1].last_x}" for b in ops]))
        # print("===================================================================")
        ret = 0
        for op in ops:
            if ret > 2:
                continue
            output = self.__get_output_banks(op)
            new_info = op.func(info, output)
            if not new_info:
                # print("Failure: op {}".format(("L" if op.action == Action.LOAD else "P") + f" {op.data[-1].tensor.name} {op.data[-1].last_x}"))
                # print(f"FMEM Status: {info.status.fmem}")
                # print("**********************************************************")
                continue

            self.__behavior_stack.append(op)
            if self.__sub_data_manager:
                print([("L" if b.action == Action.LOAD else "P") + f" {b.data[-1].tensor.name} {b.data[-1].last_x}" for b in self.__behavior_stack])
            success = self.search(new_info)
            if success:
                ret += 10 if op.action == Action.LOAD else 1
            self.__behavior_stack.pop()
            self.__data_manager.set_status(status)
            for sdm in self.__sub_data_manager:
                sdm.set_status(status)
        if self.__is_process_all(info.status):
            self.__save_best(info)
            return True
        
        if ret == 0:
            self.fail_dict.append((lx, px, self.__behavior_stack and self.__behavior_stack[-1].action == Action.PROCESS))
        return ret > 0

    def __is_process_all(self, status: Status):
        return self.__data_manager.is_process_all

    def __save_best(self, info: IntermediateInfo):
        mapping = info.mapping
        size = self.__calc_off_chip_transfer(mapping)
        if sum(size) < self.best.total_size:
            self.best.results = size
            self.best.total_size = sum(size)
            self.best.mapping = mapping
            self.best.action = copy(self.__behavior_stack)
            self.best.fmem = info.status.fmem

    def __calc_off_chip_transfer(self, mapping: MappingInfo):
        layer_info = self.__layer_info
        layer_info.mapping = mapping
        layer_info.behavior = self.actions
        in_size = layer_info.load_access_size()
        out_size = layer_info.store_access_size()
        weight_size = layer_info.weight_load_size()
        return in_size, out_size, weight_size

    @staticmethod
    def __print_results(size: Tuple[int]):
        print(f"Input Size: {size[0]:>11,d}", end=" | ")
        print(f"Output Size: {size[1]:>11,d}", end=" | ")
        print(f"Weight Size: {size[2]:>11,d}", end=" | ")
        print(f"Total Size: {sum(size):>12,d}")

    def __print_compile(self, action: List[Behavior], mapping: MappingInfo):
        print(f"Action: {[str(b) for b in action]}")
        print(f"Input : {[str(m) for m in mapping.input]}")
        print(f"Output: {[str(m) for m in mapping.output]}")

    def print_results(self):
        self.__print_results(self.best.results)

    def print_compile(self):
        self.__print_compile(self.best.action, self.best.mapping)
        print(f"FMEM Status: {self.best.fmem}\n")
