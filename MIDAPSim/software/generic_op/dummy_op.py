from __future__ import annotations

from typing import TYPE_CHECKING

from software.compiler.align_compile import NullAlign
from software.compiler.input2output import VirtualIn2Out
from software.network.op_info import OpType
from software.network.types import VTensorType

from .operator_base import OperatorBase

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData


class DummyOp(OperatorBase):
    def __init__(self, op_type="BYPASS", **kwargs):
        super(DummyOp, self).__init__(op_type=op_type, **kwargs)

    def get_op_type(self) -> OpType:
        return OpType.Dummy

    def get_alignment(self) -> Alignment:
        return NullAlign

    def get_in2out(self) -> InData2OutData:
        return VirtualIn2Out


class HostProcessOp(OperatorBase):
    def __init__(self, op_type='HOSTPROCESS', mapping='cpu', **kwargs):
        super(HostProcessOp, self).__init__(op_type=op_type, **kwargs)
        self.mapping = mapping

    def get_op_type(self) -> OpType:
        return OpType.HostProcess

    def get_alignment(self) -> Alignment:
        return NullAlign

    def get_in2out(self) -> InData2OutData:
        return VirtualIn2Out
