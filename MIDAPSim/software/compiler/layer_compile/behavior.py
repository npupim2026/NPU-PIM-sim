from __future__ import annotations

from copy import copy
from enum import Enum, auto
from typing import TYPE_CHECKING, Tuple

import attr

from .intermediate_info import IntermediateInfo, Status
from .mapping import Mapping

if TYPE_CHECKING:
    from typing import Callable, List

    from .feature_data import FeatureData

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class Action(Enum):
    PROCESS = auto()
    LOAD = auto()
    WAIT = auto()
    SYNC = auto()

@attr.s(slots=True)
class SystemBehavior(object):
    action: Action = attr.ib()
    id: Tuple[int] = attr.ib()
    @action.validator
    def __check_action(self, attribute, value):
        if not value in [Action.WAIT, Action.SYNC]:
            raise ValueError("Invalid action for system behavior")

@attr.s(slots=True)
class Behavior(object):
    action: Action = attr.ib()
    data: List[FeatureData] = attr.ib()
    func = attr.ib(init=False)

    @action.validator
    def __check_action(self, attribute, value):
        if not value in [Action.PROCESS, Action.LOAD]:
            raise ValueError("Invalid action for core behavior")

    @data.validator
    def __check_data(self, attribute, value):
        if not value:
            raise ValueError("Empty data")

    def __attrs_post_init__(self):
        self.func: Callable[[IntermediateInfo, List[FeatureData]], IntermediateInfo] = (
            self.__load if self.is_load() else self.__process
        )

    def __load(self, info: IntermediateInfo, output: List[FeatureData]):
        assert not output
        status : Status = info.status
        mapping = info.mapping

        copied_status = copy(status)
        copied_mapping = copy(mapping)
        for d in self.data:
            bank = copied_status.fmem.write(d)
            if bank is None:
                return None
            copied_mapping.input.append(Mapping(bank, d))
            copied_status.last_load[d.tensor] = d.last_x
            copied_status.last_process[d.tensor] = min(d.pivot, copied_status.last_process[d.tensor])
        return IntermediateInfo(copied_status, copied_mapping)

    def __process(self, info: IntermediateInfo, output: List[FeatureData]):
        status = info.status
        mapping = info.mapping

        copied_status = copy(status)
        copied_mapping = copy(mapping)
        for d in output:
            if copied_status.fmem.is_saved(d):
                continue
            bank = copied_status.fmem.write(d)
            if bank is None:
                return None
            copied_mapping.output.append(Mapping(bank, d))
        copied_status.fmem.invalidate(self.data)
        for d in self.data:
            copied_status.last_process[d.tensor] = d.last_x
        return IntermediateInfo(copied_status, copied_mapping)

    @property
    def num_banks(self) -> int:
        return len(self.data)

    def is_load(self) -> bool:
        return self.action == Action.LOAD

    def is_process(self) -> bool:
        return self.action == Action.PROCESS

    def __str__(self) -> str:
        return f"({self.action.name[:1]} {self.num_banks} {' '.join(['<' + ','.join([f.tensor.name, str(f.pivot), str(f.last_x)]) + '>' for f in self.data])})"
