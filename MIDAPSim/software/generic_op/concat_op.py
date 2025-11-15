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


class ConcatOp(OperatorBase):
    def __init__(self, op_type="Concat", axis=1, concat_info=None, **kwargs):
        super(ConcatOp, self).__init__(op_type=op_type, **kwargs)
        if not isinstance(concat_info, list):
            raise ValueError("concat_info must be given as a size list")
        self.axis = axis
        self.size_info = [sum(concat_info[:i]) for i in range(len(concat_info))]

    def __repr__(self):
        return super(ConcatOp, self).__repr__() + "offset information: {}\n".format(
            self.size_info
        )

    def get_in2out(self) -> InData2OutData:
        return DefaultIn2Out

    def tensor_to_midap_tensor(self):
        if self.order == "NCHW":
            super(ConcatOp, self).tensor_to_midap_tensor()
            # axis_translate_arr = [-1, 2, 1, 0]  # NCHW
            axis_translate_arr = [-1, 2, 0, 1]  # NCHW #FIXME Temporally, we support only x-axis concatenation
            self.axis = axis_translate_arr[self.axis]

    def get_op_type(self) -> OpType:
        return OpType.Concat

    def get_alignment(self) -> Alignment:
        return OutAlign

    @property
    def vtype(self):
        return VTensorType.OutputLinear
