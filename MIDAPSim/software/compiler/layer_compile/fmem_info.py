from __future__ import annotations

from collections import deque
from copy import copy
from typing import TYPE_CHECKING

from config import cfg

from .feature_data import FeatureData

if TYPE_CHECKING:
    from typing import List


__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class FMEMInfo:
    def __init__(self):
        self.__num_banks = cfg.MIDAP.FMEM.NUM
        self.__bank_size = cfg.MIDAP.FMEM.NUM_ENTRIES
        self.__available_banks = deque([i for i in range(self.num_banks)])
        self.__data = [FeatureData() for _ in range(self.num_banks)]

        self.__fixed_banks = set()

    @property
    def bank_size(self):
        return self.__bank_size

    @property
    def num_banks(self):
        return self.__num_banks

    def set_stationary(self, data: List[FeatureData]):
        self.__fixed_banks = set(data)

    def invalidate(self, data: List[FeatureData]):
        assert all([d in self.__data for d in data])
        for d in data:
            if d in self.__fixed_banks:
                continue
            bank = self.__data.index(d)
            assert self.__data[bank] in data
            self.__data[bank] = FeatureData()
            self.__available_banks.append(bank)

    def _pop_empty_bank(self):
        if self.__available_banks:
            return self.__available_banks.popleft()
        return None

    def write(self, data: FeatureData):
        assert data.size <= self.__bank_size
        bank = self._pop_empty_bank()
        if bank is not None:
            self.__data[bank] = data
            return bank
        return None

    def read(self, num: int):
        assert num < self.__num_banks
        return self.__data[num]

    def __copy__(self):
        new_fmem = FMEMInfo()
        new_fmem.__available_banks = copy(self.__available_banks)
        new_fmem.__data = copy(self.__data)
        new_fmem.__fixed_banks = copy(self.__fixed_banks)
        return new_fmem

    def __iter__(self):
        return iter(self.__data)

    def is_saved(self, data: FeatureData):
        return data in self.__data

    @property
    def num_available_banks(self):
        return len(self.__available_banks)

    def __repr__(self) -> str:
        return f'{["O" if d.tensor.name == "empty" else f"({d.tensor.name} [{d.pivot}, {d.last_x}])" for d in self.__data]}'
