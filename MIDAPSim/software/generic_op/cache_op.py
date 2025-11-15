from __future__ import annotations

from typing import TYPE_CHECKING

from software.compiler.align_compile import OutAlign
from software.compiler.input2output import DefaultIn2Out
from software.network.op_info import OpType
from software.network.types import VTensorType

from .operator_base import OperatorBase

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData


class CacheOp(OperatorBase):
    def __init__(self, op_type="Cache", shape=(0, 0, 0), write_offset=(0, 0, 0), write_shape=(0, 0, 0), tag='', **kwargs):
        super(CacheOp, self).__init__(op_type=op_type, **kwargs)
        self.shape = shape
        self.write_offset = write_offset
        self.write_shape = write_shape
        self.tag = tag

    def __repr__(self):
        return super(CacheOp, self).__repr__() + "Max size: {}\n".format(
            self.shape
        )

    def get_in2out(self) -> InData2OutData:
        return DefaultIn2Out

    def tensor_to_midap_tensor(self):
        if self.order == "NCHW":
            super(CacheOp, self).tensor_to_midap_tensor()
            self.shape = (self.shape[1], self.shape[2], self.shape[0])
            self.write_offset = (self.write_offset[1], self.write_offset[2], self.write_offset[0])
            self.write_shape = (self.write_shape[1], self.write_shape[2], self.write_shape[0])

    def get_op_type(self) -> OpType:
        return OpType.Concat

    def get_alignment(self) -> Alignment:
        return OutAlign

    @property
    def vtype(self):
        return VTensorType.OutputWMEM
