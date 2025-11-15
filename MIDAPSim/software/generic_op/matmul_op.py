from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from software.compiler.align_compile import MatMulAlign
from software.compiler.input2output import ConvPoolIn2Out
from software.network.op_info import OpType

from .convpool_op_base import ConvPoolOpBase

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData


class MatMulOp(ConvPoolOpBase):
    def __init__(
        self,
        op_type='MatMul',
        vec_len=0,
        **kwargs,
    ):
        super(MatMulOp, self).__init__(op_type=op_type, **kwargs)
        self.vec_len = vec_len

    def get_in2out(self) -> InData2OutData:
        return ConvPoolIn2Out

    def get_op_type(self) -> OpType:
        if self.type == 'MatMul':
            return OpType.MatMul
        return OpType.MatMulTrans

    def get_alignment(self) -> Alignment:
        return MatMulAlign

    def get_macs(self):  # Overrided
        return self.output_tensor[0].size * self.vec_len

    def flip_operation(self, flip):
        # Not implemented
        raise NotImplementedError
