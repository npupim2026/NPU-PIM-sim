from __future__ import annotations

import logging
from typing import List

import attr
import numpy as np
from config import cfg
from logger import init_logger
from software.network.tensor import Tensor
from software.network.types import TensorType

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


logger = init_logger("Min Policy", logging.INFO)


@attr.s(slots=True, hash=True)
class FeatureData(object):
    tensor: Tensor = attr.ib(default=Tensor("empty", np.ndarray(0), TensorType.Empty))
    pivot: int = attr.ib(default=int(0))
    width: int = attr.ib(default=int(0))
    plane_size: int = attr.ib(default=int(0))
    size: int = attr.ib(default=int(0))

    @classmethod
    def initialize(cls, tensor: Tensor, x_pivot: int, x_width: int):
        tensor = tensor
        pivot = x_pivot
        width = x_width
        plane_size = tensor.yz_plane_size
        size = plane_size * width
        return cls(tensor, pivot, width, plane_size, size)

    @property
    def last_x(self) -> int:
        return self.pivot + self.width

    def overlap(self, min: int, max: int) -> bool:
        return max > self.pivot and self.last_x > min


class FeatureFragment(List[FeatureData]):
    @staticmethod
    def _init_reversed(num_planes: int, head: int, tensor: Tensor) -> List[FeatureData]:
        data = []
        for offset in range(tensor.width, head, -num_planes):
            width = min(num_planes, offset - head)
            data.append(FeatureData.initialize(tensor, offset - width, width))
        return data

    @staticmethod
    def _init_ordered(num_planes: int, head: int, tensor: Tensor) -> List[FeatureData]:
        data = []
        for offset in range(head, tensor.width, num_planes):
            width = min(num_planes, tensor.width - offset)
            data.append(FeatureData.initialize(tensor, offset, width))
        return data

    @classmethod
    def initialize(cls, tensor: Tensor, head: int = 0, reverse: bool = False):
        num_planes = cfg.MIDAP.FMEM.NUM_ENTRIES // tensor.yz_plane_size
        if reverse:
            data = cls._init_reversed(num_planes, head, tensor)
        else:
            data = cls._init_ordered(num_planes, head, tensor)
        return cls(data)
