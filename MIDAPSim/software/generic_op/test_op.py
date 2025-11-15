from __future__ import annotations

from typing import TYPE_CHECKING

from software.compiler.align_compile import NullAlign
from software.compiler.input2output import DefaultIn2Out
from software.network.op_info import OpType
from software.network.types import VTensorType

from .operator_base import OperatorBase

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData


class TestOp(OperatorBase):
    def __init__(self, op_type="Test", behavior=[], **kwargs):
        super(TestOp, self).__init__(op_type=op_type, **kwargs)
        self.behavior = behavior

    def get_in2out(self) -> InData2OutData:
        return DefaultIn2Out

    def get_op_type(self) -> OpType:
        return OpType.Test

    def get_alignment(self) -> Alignment:
        return NullAlign

    @property
    def test_code(self):
        return self.behavior
