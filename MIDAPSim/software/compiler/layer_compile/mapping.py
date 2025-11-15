from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import attr

if TYPE_CHECKING:
    from typing import List, Tuple

    from .feature_data import FeatureData

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


@attr.s(slots=True)
class Mapping:
    bank: int = attr.ib()
    data: FeatureData = attr.ib()

    def __str__(self) -> str:
        return str(self.to_tuple())

    def to_tuple(self) -> Tuple:
        return (self.bank, (self.data.pivot, self.data.last_x))


@attr.s(slots=True)
class MappingInfo:
    input: List[Mapping] = attr.ib(default=attr.Factory(list))
    output: List[Mapping] = attr.ib(default=attr.Factory(list))

    @property
    def last_out_x(self) -> int:
        return self.output[-1].data.last_x if len(self.output) > 0 else 0

    def __copy__(self):
        new_info = MappingInfo()
        new_info.input = copy(self.input)
        new_info.output = copy(self.output)
        return new_info
