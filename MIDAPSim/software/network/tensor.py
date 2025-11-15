from __future__ import annotations
from software.network.quant_info import TensorQuantInfo, QuantType

import numpy as np
from config import cfg

from .types import TensorType


__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class Tensor(object):  # Substantial Tensors
    __slots__ = (
        "__name",
        "__data",
        "__flip_x",
        "__type",
        "__shared",
        "__init_shape",
        "__quant_info",
    )

    def __init__(self, name, data, type, quant_info=None):
        self.__name: str = name
        self.__data: np.ndarray = data
        self.__type: TensorType = type
        self.__flip_x: bool = False
        self.__shared: bool = False
        self.__quant_info: TensorQuantInfo = quant_info
        self.__init_shape = self.shape

    def __repr__(self):
        return f"<Tensor {self.__name} ({self.shape} {self.__type} Flipped: {self.__flip_x})>"

    @property
    def is_input_tensor(self):
        return self.__type == TensorType.In

    @property
    def is_constant_tensor(self):
        return self.__type == TensorType.Constant

    @property
    def is_shared_tensor(self):
        return self.__shared

    @property
    def name(self):
        return self.__name

    @property
    def data(self):
        return self.__data

    @property
    def type(self):
        return self.__type

    @property
    def shared(self):
        return self.__shared

    @type.setter
    def type(self, type):
        self.__type = type

    @data.setter
    def data(self, data):
        self.__data = data

    @shared.setter
    def shared(self, shared):
        self.__shared = shared

    @property
    def quant_info(self):
        return self.__quant_info

    @quant_info.setter
    def quant_info(self, info: TensorQuantInfo):
        self.__quant_info = info

    @property
    def flip_x(self):
        return self.__flip_x

    @flip_x.setter
    def flip_x(self, flip_x: bool = True):
        if self.__flip_x ^ flip_x:
            self.__data = np.flip(self.__data, axis=1)
        self.__flip_x = flip_x

    @property
    def shape(self):
        return self.__data.shape

    @property
    def is_quantized(self):
        return (
            self.__quant_info is not None
            and self.__quant_info.type != QuantType.Default
        )

    @property
    def yz_plane_size(self) -> int:
        import numpy as np

        return np.prod(self.shape[2:])

    @property
    def num_yz_planes(self) -> int:
        return cfg.MIDAP.FMEM.NUM_ENTRIES // self.yz_plane_size

    @property
    def size_per_bank(self) -> int:
        return self.yz_plane_size * self.num_yz_planes

    @property
    def total_size(self) -> int:
        import numpy as np

        # Screwed up by the multi-frame update...
        if self.is_constant_tensor:
            return np.prod(self.shape)
        return np.prod(self.shape[1:])

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[2]

    @property
    def init_shape(self):
        return self.__init_shape

    @property
    def init_size(self) -> int:
        return np.prod(self.init_shape[1:])
