from __future__ import annotations
from enum import Enum, auto

import attr

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"

class ComputeType(Enum):
    StdConv = auto()
    DWConv = auto()
    Pool = auto()
    Elwise = auto()
    WeightedSum = auto()
    MatMul = auto()
    MatMulTrans = auto()

@attr.s(slots=True, init=True)
class WMEMInfo(object):
    filter_name: str = attr.ib(default=None)
    compute_type : ComputeType = attr.ib(default = attr.Factory(ComputeType))
    load_filter_once: bool = attr.ib(default=False)
    filter_group_size: int = attr.ib(default=1)
    filter_group_offset: int = attr.ib(default=1)
    prepare_info: str = attr.ib(default=None)  # layer name
    prepared: bool = attr.ib(default=False)
