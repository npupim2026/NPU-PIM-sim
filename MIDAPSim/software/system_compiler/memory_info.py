from __future__ import annotations

import logging
from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING, DefaultDict, Dict, List

import attr
import numpy as np
from config import cfg
from logger import init_logger
from software.system_compiler.dependency_info import DependencyInfo
from software.system_compiler.prefetch_info import PrefetchInfo, TransferTrace

if TYPE_CHECKING:
    from typing import Tuple

    import numpy.typing as npt

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"

logger = init_logger("Memory Info", logging.INFO)


class MemoryType(Enum):
    Shared = 0
    Input = 1
    Output = 2
    Constant = 3
    Temporal = 4
    SPM = 4


@attr.s(slots=True, init=True)
class LocalMemoryInfo(object):
    address_dict: OrderedDict[str, int] = attr.ib(
        default=attr.Factory(OrderedDict)
    )  # relative address for its temporal buffer
    write_mem_pivot_dict: OrderedDict[str, int] = attr.ib(
        default=attr.Factory(OrderedDict)
    )  # Update off-chip write criteria for each tensor
    required_size: int = attr.ib(default=int(0))


@attr.s(slots=True, init=True)
class SPMInfo(object):
    num_banks: int = attr.ib(default=1)
    bank_size: int = attr.ib(default=0)
    data: npt.ArrayLike = attr.ib(default=np.zeros(0))
    bank_prefetch_info: Dict[int, List[PrefetchInfo]] = attr.ib(
        default=attr.Factory(DefaultDict)
    )

    @property
    def size(self):
        return self.num_banks * self.bank_size

    @property
    def in_use(self):
        return self.size > 0


@attr.s(slots=True, init=True)
class MemoryInfo(object):
    spm_info: SPMInfo = attr.ib(default=attr.Factory(SPMInfo))
    data_shared_constants: npt.ArrayLike = attr.ib(default=np.zeros(0))
    data_shared_feature: npt.ArrayLike = attr.ib(default=np.zeros(0))
    data_input_feature: npt.ArrayLike = attr.ib(default=np.zeros((1, 0)))
    data_output_feature: npt.ArrayLike = attr.ib(default=np.zeros((1, 0)))
    shared_address_dict: OrderedDict[str, Tuple[int, int]] = attr.ib(
        default=attr.Factory(OrderedDict)
    )
    local_memory_info_dict: OrderedDict[int, LocalMemoryInfo] = attr.ib(
        default=attr.Factory(OrderedDict)
    )

    def get_sim_memory_info(self, core_id):
        default = OrderedDict()
        default[MemoryType.Shared] = dict(
            data=self.data_shared_feature, offset=cfg.DRAM.OFFSET.SHARED
        )
        default[MemoryType.Input] = dict(
            data=self.data_input_feature, offset = cfg.DRAM.OFFSET.INPUT
        )
        default[MemoryType.Output] = dict(
            data=self.data_output_feature, offset = cfg.DRAM.OFFSET.OUTPUT
        )
        default[MemoryType.Constant] = dict(
            data=self.data_shared_constants, offset=cfg.DRAM.OFFSET.WEIGHT_BIAS
        )

        dtype = self.data_shared_constants.dtype
        if self.spm_info.in_use:
            if self.spm_info.data.size != self.spm_info.size:
                self.spm_info.data = np.zeros(self.spm_info.size, dtype=dtype)
            default[MemoryType.SPM] = dict(
                data=self.spm_info.data, offset=cfg.DRAM.OFFSET.BUFFER
            )
        else:
            offset = int(cfg.DRAM.OFFSET.BUFFER)
            for id, info in self.local_memory_info_dict.items():
                if id == core_id:
                    break
                offset += info.required_size
            default[MemoryType.Temporal] = dict(
                data=np.zeros(self.local_memory_info_dict[core_id].required_size, dtype = dtype),
                offset=int(offset),
            )
        return default

    def get_address_dict(self, core_id):
        addr_dict = OrderedDict()
        addr_dict.update(self.shared_address_dict)
        for name, buf_addr in self.local_memory_info_dict[core_id].address_dict.items():
            if name not in addr_dict:
                addr_dict[name] = (MemoryType.Temporal.value, buf_addr)
        return addr_dict

    def generate_prefetch_trace(
        self, core_idx, dep_info: DependencyInfo
    ) -> OrderedDict[int, List[TransferTrace]]:
        trace = OrderedDict()
        if not self.spm_info.in_use:
            return trace
        sync_dict = dep_info.get_core_layer_id_info(core_idx - 1)
        dep_dict = dep_info.get_core_dependency_info(core_idx - 1)
        for bank_id in range(self.spm_info.num_banks):
            bank_trace = []
            for prefetch in self.spm_info.bank_prefetch_info[bank_id]:
                wait_list = dep_dict[prefetch.name]
                sync_id = sync_dict[prefetch.name]
                for wait_id in wait_list:
                    bank_trace.append(TransferTrace.Wait(wait_id))
                src = self.shared_address_dict[prefetch.src]
                dst = self.shared_address_dict[prefetch.dst]
                bank_trace.append(TransferTrace.Transfer(src, dst, prefetch.size))
                bank_trace.append(TransferTrace.Sync(sync_id))
            trace[bank_id] = bank_trace
        return trace
