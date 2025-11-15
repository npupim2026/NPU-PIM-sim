from __future__ import annotations

from typing import TYPE_CHECKING

from software.compiler.align_compile import ArithAlign
from software.network.op_info import OpType

from .arithmetic_op import ArithmeticOp

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment


class SumOp(ArithmeticOp):
    def __init__(self, **kwargs):
        # TODO: add other arguments for the upsampling operation
        # Upsampling ratio, ...
        super(SumOp, self).__init__(op_type="Sum", **kwargs)

    def get_op_type(self) -> OpType:
        from config import cfg
        if cfg.MIDAP.CONTROL_STRATEGY.WEIGHTED_SUM:
            return OpType.WeightedSum
        else:
            return OpType.Sum

    def get_alignment(self) -> Alignment:
        return ArithAlign
