from __future__ import annotations

from typing import TYPE_CHECKING
from software.compiler.input2output import DefaultIn2Out
from .operator_base import OperatorBase

if TYPE_CHECKING:
    from software.compiler.input2output import InData2OutData


class ArithmeticOp(OperatorBase):
    def __init__(self, broadcast=False, activation=None, **kwargs):
        # TODO: add other arguments for the upsampling operation
        # Upsampling ratio, ...
        super(ArithmeticOp, self).__init__(**kwargs)
        self.__broadcast = broadcast
        self.activation = activation

    def get_in2out(self) -> InData2OutData:
        return DefaultIn2Out

    @property
    def broadcast(self):
        return self.__broadcast
