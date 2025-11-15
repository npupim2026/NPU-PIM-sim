from __future__ import annotations

from typing import TYPE_CHECKING

from software.compiler.align_compile import ArithAlign
from software.compiler.input2output import ConvPoolIn2Out, GlobalPoolIn2Out
from software.network.op_info import OpType

from .convpool_op_base import ConvPoolOpBase

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData


class PoolOp(ConvPoolOpBase):
    def __init__(
        self,
        op_type="Pool",
        pool_type=None,
        global_pooling=False,
        **kwargs
        ):
        super(PoolOp, self).__init__(op_type=op_type, **kwargs)
        self.global_pooling: bool = global_pooling
        if pool_type is not None:
            self.type = pool_type

    def get_in2out(self) -> InData2OutData:
        if self.global_pooling:
            return GlobalPoolIn2Out
        return ConvPoolIn2Out

    def get_op_type(self) -> OpType:
        if self.global_pooling:
            return OpType.GlobalPool
        if self.type == "MaxPool":
            return OpType.MaxPool
        if self.type == "AveragePool":
            return OpType.AvgPool
        raise ValueError(f"Type cannot be {self.type}")

    def get_alignment(self) -> Alignment:
        return ArithAlign
