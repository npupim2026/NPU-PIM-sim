from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from functools import reduce
from config import cfg

from logger import init_logger

from .tensor import Tensor
from .types import TensorType, VTensorType
from .quant_info import QuantType, TensorQuantInfo

if TYPE_CHECKING:
    from typing import List, Tuple

    from software.generic_op.operator_base import OperatorBase

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"

logger = init_logger("Layer", logging.INFO)


class VirtualTensor(object):
    __slots__ = ("__tensor", "__shape", "__type", "__offset", "__scale", "__init_shape")

    def __init__(
        self, tensor: Tensor, shape=None, tensor_type=None, scale=None, offset=None
    ):
        self.__tensor = tensor
        self.__shape: Tuple(int, int, int, int) = (
            tensor.shape if shape is None else shape
        )
        self.__type: VTensorType = (
            tensor_type if tensor_type is not None else VTensorType.Default
        )
        self.__scale: Tuple(int, int, int) = (
            (1, 1, 1) if scale is None else scale
        )  # WHC
        self.__offset: Tuple(int, int, int) = (
            (0, 0, 0) if offset is None else offset
        )  # WHC
        self.__init_shape: Tuple(int, int, int, int) = self.__shape

    def __repr__(self):
        return f"<Virtual Tensor: Orig Tensor {self.__tensor}, -> tensor type {self.__type} / scale {self.__scale} / offset {self.__offset} / shape {self.__shape})>"

    def __copy__(self):
        tensor = self.tensor
        shape = self.shape
        tensor_type = self.type
        scale = self.scale
        offset = self.offset
        result = VirtualTensor(tensor, shape, tensor_type, scale, offset)
        result.__init_shape = self.init_shape
        return result

    @classmethod
    def from_op(cls, op: OperatorBase, in_tensors: List[VirtualTensor]):
        if op.type == 'BYPASS' and in_tensors:
            return in_tensors[0]
        elif cfg.MODEL.ALLOW_ABSTRACT_DATA and op.vtype in [VTensorType.InputLinear, VTensorType.InputValid]:
            vtype = op.vtype
            shape = op.output_tensor.shape
            scale = op.vscale
            offset = op.voffset
            tensor = in_tensors[0].tensor
            return cls(tensor, shape, vtype, scale, offset)
        else:
            if op.type == 'CONST':
                ttype = TensorType.Constant
            elif op.is_first_tensor:
                ttype = TensorType.In
            elif op.next:
                ttype = TensorType.Temporal
            else:
                ttype = TensorType.Out
            quant_info = None
            if op.output_scale is not None:
                quant_info = TensorQuantInfo(type=QuantType.Symm, scale=op.output_scale)
            tensor = Tensor(op.tag if op.type == "Cache" else op.name, op.output_tensor, ttype, quant_info)
            if cfg.MODEL.ALLOW_ABSTRACT_DATA and op.type == "Concat":
                output_scale = reduce(lambda acc, cur: max(acc, cur.scale) if cur is not None else acc, [vtensor.tensor.quant_info for vtensor in in_tensors], 0)
                if output_scale:
                    tensor.quant_info = TensorQuantInfo(type=QuantType.Symm, scale=op.output_scale)
                for voffset, vtensor in zip(op.size_info, in_tensors):
                    new_offset = [0, 0, 0]
                    new_offset[op.axis] = voffset
                    vtensor.offset = tuple(new_offset)
                    vtensor.tensor = tensor
            elif cfg.MODEL.ALLOW_ABSTRACT_DATA and op.type == "Cache":
                tensor.shared = True    # Allocate this tensor on the shared section of DRAM
                for vtensor in in_tensors:
                    vtensor.offset = tuple(op.write_offset)
                    vtensor.tensor = tensor
                    vtensor.type = VTensorType.OutputWMEM
            return cls(tensor)

    @property
    def is_input_tensor(self):
        return self.tensor.is_input_tensor

    @property
    def name(self):
        return self.__tensor.name

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, type):
        self.__type = type

    @property
    def tensor(self):
        return self.__tensor

    @tensor.setter
    def tensor(self, tensor):
        self.__tensor = tensor

    @property
    def data(self):
        return self.tensor.data

    @data.setter
    def data(self, data):
        if self.__type != VTensorType.Default:
            raise ValueError(
                f"Not allowed data setup for the virtualized tensor: {self}"
            )
        self.tensor.data = data
        self.shape = data.shape

    @property
    def orig_shape(self):
        return self.tensor.shape

    @property
    def shape(self):
        if self.__type == VTensorType.Default:
            return self.tensor.shape
        else:
            return self.__shape

    @shape.setter
    def shape(self, shape):
        self.__shape = shape

    @property
    def scale(self):
        return self.__scale

    @property
    def offset(self):
        return self.__offset

    @offset.setter
    def offset(self, offset: Tuple(int, int, int)):
        self.__type = VTensorType.OutputLinear
        self.__offset = offset

    @property
    def flip_x(self) -> bool:
        return self.__tensor.flip_x

    @flip_x.setter
    def flip_x(self, flip_x: bool = True) -> bool:
        self.__tensor.flip_x = flip_x

    @property
    def yz_plane_size(self) -> int:
        return self.__tensor.yz_plane_size

    @property
    def num_yz_planes(self) -> int:
        return self.__tensor.num_yz_planes

    @property
    def size_per_bank(self) -> int:
        return self.__tensor.size_per_bank

    @property
    def total_size(self) -> int:
        return self.__tensor.total_size

    @property
    def width(self) -> int:
        return self.__tensor.width

    @property
    def vwidth(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.__tensor.height

    @property
    def orig_init_shape(self) -> Tuple(int, int, int, int):
        return self.tensor.init_shape

    @property
    def init_shape(self) -> Tuple(int, int, int, int):
        return self.__init_shape

    def get_input_x(self, in_x):
        return (in_x - self.offset[0]) * self.scale[0]

    def get_output_x(self, out_x):
        return out_x + self.offset[0]

    def to_torch_tensor(self):
        import numpy as np
        import torch
        import torch.nn.functional as F

        logger.debug(f"to_torch_tensor() Called: {self}")
        it = self.tensor.data
        xo, yo, zo = self.offset
        xs, ys, zs = self.scale
        _, x, y, z = self.shape
        if any(
            [zo > 0, yo > 0, not zs == 1, not z == self.orig_shape[-1], not xs == ys]
        ):
            raise ValueError(f"This tensor cannot be supported: {self}")
        data = it[:, xo:, yo:, zo:].transpose(0, 3, 2, 1)
        data = torch.from_numpy(data.astype(np.float32))
        if xs > 1 or ys > 1:
            if self.type == VTensorType.InputLinear:
                data = torch.nn.UpsamplingNearest2d(scale_factor=ys)(data)
            elif self.type == VTensorType.InputValid:
                w = data.new_zeros(ys, xs)
                w[0, 0] = 1
                data = F.conv_transpose2d(
                    data,
                    w.expand(data.size(1), 1, ys, xs),
                    stride=xs,
                    groups=data.size(1),
                )
        data = data[:, :z, :y, :x]
        return data
