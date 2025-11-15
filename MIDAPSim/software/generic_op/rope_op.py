from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from software.compiler.align_compile import ArithAlign
from software.compiler.input2output import DefaultIn2Out
from software.network.op_info import OpType

from .operator_base import OperatorBase

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData


class RoPEOp(OperatorBase):
    def __init__(
        self,
        op_type='RoPE',
        dim=256,
        base=10000,
        input_pos=0,
        input_len=1,
        **kwargs,
    ):
        super(RoPEOp, self).__init__(op_type=op_type, **kwargs)
        self.base = base
        self.dim = dim
        self.input_pos = input_pos
        self.input_len = input_len

    def get_in2out(self) -> InData2OutData:
        return DefaultIn2Out

    def get_op_type(self) -> OpType:
        return OpType.RoPE

    def get_alignment(self) -> Alignment:
        return ArithAlign

    def get_macs(self):  # Overrided
        return self.output_tensor[0].size * 2

    def flip_operation(self, flip):
        # Not implemented
        raise NotImplementedError
