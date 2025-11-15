from __future__ import annotations

from typing import TYPE_CHECKING

from software.compiler.align_compile import ArithAlign
from software.network.op_info import OpType

from .arithmetic_op import ArithmeticOp

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment


class MulOp(ArithmeticOp):
    def __init__(self, **kwargs):
        # TODO: add other arguments for the upsampling operation
        # Upsampling ratio, ...
        super(MulOp, self).__init__(op_type="Mul", **kwargs)

    def get_op_type(self) -> OpType:
        return OpType.Mul

    def get_alignment(self) -> Alignment:
        return ArithAlign
