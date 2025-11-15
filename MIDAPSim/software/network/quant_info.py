from __future__ import annotations

from enum import Enum, auto
from software.network.types import TensorType
from typing import Dict, List, OrderedDict, Tuple

import attr

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


class QuantType(Enum):  # 21/08/03: Not in use
    Default = auto()
    Symm = auto()
    Linear = auto()
    LogScale = auto()


@attr.s(slots=True, init=True)
class TensorQuantInfo(object):
    type: QuantType = attr.ib(default=QuantType.Default)
    bits: int = attr.ib(default=8)
    scale: float = attr.ib(
        default=1.0
    )  # 21/08/03: Use this term only for current MIDAP Scheme
    offset: int = attr.ib(default=0)


@attr.s(slots=True, init=True)
class LayerQuantInfo(object):
    type: QuantType = attr.ib(default=QuantType.Default)
    scale_32b_to_16b: float = attr.ib(default=None) # Datapath Output (32B) -> 16B (No truncate?)
    scale_16b_to_8b: float = attr.ib(default=None) # LUT/Activation Output (16B) -> 8B (w/ Truncate)
    bias_scale: float = attr.ib(default=1.0)
    activation_lut = attr.ib(default=None)

    @property
    def scales(self):
        return (self.scale_32b_to_16b, self.scale_16b_to_8b)

    def set_lut(self, name, data):
        import numpy as np
        from software.network.tensor import Tensor
        new_data = np.zeros([data.shape[0], data.shape[1] * 2], dtype = np.int8)
        for i in range(2):
            new_data[:,2*i] = data[:,i].astype(np.int8)
            new_data[:,2*i + 1] = np.right_shift(data[:,i], 8).astype(np.int8)
        self.activation_lut = Tensor(name, new_data, type = TensorType.Constant)