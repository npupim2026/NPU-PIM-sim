from __future__ import annotations

from typing import TYPE_CHECKING

from software.compiler.align_compile import ArithAlign
from software.compiler.input2output import VirtualIn2Out
from software.network.op_info import OpType
from software.network.types import VTensorType

from .operator_base import OperatorBase

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData


class Crop(OperatorBase):
    def __init__(self, op_type="Crop", crop_x=None, crop_y=None, crop_z=None, **kwargs):
        OperatorBase.__init__(self, op_type=op_type, **kwargs)
        if crop_x is not None:
            if not isinstance(crop_x, list) and not isinstance(crop_x, tuple):
                raise ValueError("crop_x should be tuple or list")
            if len(crop_x) != 2:
                raise ValueError("len(crop_x) should be 2")
            if crop_x[1] > 0:
                raise ValueError("crop_x[1] should be zero or negative")
        else:
            crop_x = tuple((0, 0))
        if crop_y is not None:
            if not isinstance(crop_y, list) and not isinstance(crop_y, tuple):
                raise ValueError("crop_y should be tuple or list")
            if len(crop_y) != 2:
                raise ValueError("len(crop_y) should be 2")
            if crop_y[1] > 0:
                raise ValueError("crop_y[1] should be zero or negative")
        else:
            crop_y = tuple((0, 0))
        if crop_z is not None:
            if not isinstance(crop_z, list) and not isinstance(crop_z, tuple):
                raise ValueError("crop_z should be tuple or list")
            if len(crop_z) != 2:
                raise ValueError("len(crop_y) should be 2")
            if crop_z[1] > 0:
                raise ValueError("crop_z[1] should be zero or negative")
        else:
            crop_z = tuple((0, 0))
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_z = crop_z

    def get_in2out(self) -> InData2OutData:
        return VirtualIn2Out

    def get_op_type(self) -> OpType:
        return OpType.Crop

    def get_alignment(self) -> Alignment:
        return ArithAlign

    @property
    def voffset(self):
        return (self.crop_y[0], self.crop_x[0], self.crop_z[0])

    @property
    def vtype(self):
        return VTensorType.InputLinear
