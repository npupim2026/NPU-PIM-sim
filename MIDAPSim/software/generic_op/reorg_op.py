from __future__ import annotations

from typing import TYPE_CHECKING

from software.compiler.align_compile import ArithAlign
from software.compiler.input2output import VirtualIn2Out
from software.network.op_info import OpType
from software.network.types import VTensorType

from .convpool_op_base import ConvPoolOpBase

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData


class ReorgOp(ConvPoolOpBase):
    def __init__(self, op_type="Reorg", **kwargs):
        # TODO: add other arguments for the upsampling operation
        # Upsampling ratio, ...
        ConvPoolOpBase.__init__(self, op_type=op_type, **kwargs)
        # Not yet supported

    def get_in2out(self) -> InData2OutData:
        return VirtualIn2Out

    def get_op_type(self) -> OpType:
        return OpType.Reorg

    def get_alignment(self) -> Alignment:
        return ArithAlign

    @property
    def vscale(self):
        return (self.k_w, self.k_h, 1)

    @property
    def vtype(self) -> str:
        return VTensorType.OutputReorg
