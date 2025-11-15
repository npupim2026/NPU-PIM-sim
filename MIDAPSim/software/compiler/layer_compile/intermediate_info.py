from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Dict, List

import attr
from torch.functional import Tensor

from .mapping import Mapping, MappingInfo

if TYPE_CHECKING:
    from .fmem_info import FMEMInfo


@attr.s(slots=True)
class Status:
    fmem: FMEMInfo = attr.ib(default=None)
    last_load: Dict[Tensor, int] = attr.ib(default=attr.Factory(dict))
    last_process: Dict[Tensor, int] = attr.ib(default=attr.Factory(dict))

    def __copy__(self):
        new_status = Status()
        new_status.fmem = copy(self.fmem)
        new_status.last_load = copy(self.last_load)
        new_status.last_process = copy(self.last_process)
        return new_status


@attr.s(slots=True)
class IntermediateInfo:
    status: Status = attr.ib()
    mapping: MappingInfo = attr.ib(default=attr.Factory(MappingInfo))

    @classmethod
    def from_fmem(cls, fmem: FMEMInfo, vtensors: List[VirtualTensor]):
        mapping = MappingInfo()
        last_load = {vtensor.tensor : vtensor.offset[0] for vtensor in vtensors}
        last_process = copy(last_load)
        tensors = [vtensor.tensor for vtensor in vtensors]
        for bank, d in enumerate(fmem):
            if d.tensor in tensors:
                mapping.input.append(Mapping(bank, d))
                last_load[d.tensor] = d.last_x if last_load[d.tensor] < d.last_x else last_load[d.tensor]
        mapping.input = sorted(mapping.input, key=(lambda m: (m.data.pivot, tensors.index(m.data.tensor))))
        return cls(Status(fmem, last_load, last_process), mapping)
