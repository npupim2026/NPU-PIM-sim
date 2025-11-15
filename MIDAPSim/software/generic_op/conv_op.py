from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from software.compiler.align_compile import (
    DWConvAlign,
    GEMMAlign,
    NullAlign,
    StdConvAlign,
)
from software.compiler.input2output import ConvPoolIn2Out
from software.network.op_info import OpType

from .convpool_op_base import ConvPoolOpBase

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData


class ConvOp(ConvPoolOpBase):
    def __init__(
        self,
        bn=None,
        dilation=1,
        group=1,
        op_type="Conv",
        **kwargs,
    ):
        super(ConvOp, self).__init__(op_type=op_type, **kwargs)
        if self.weight is None:
            raise ValueError("weight tensor(np.array) must be set")
        # bn = [gamma, beta, mu, sigma] [Warning: bn[3] (sigma) should be merged with epsilon, like as sqrt(sigma^2 + epsilon)
        self.bn = bn
        self.__dilation = dilation
        self.group = group
        if not any([group == 1, group == self.weight.shape[0], op_type == "UpsampleBilinear"]):
            raise ValueError("Normal Conv or Depthwise Conv is only supported.")
        if group > 1 and op_type != "UpsampleBilinear":
            self.type = "Depthwise"

    @property
    def dilation(self):
        return self.__dilation

    def get_in2out(self) -> InData2OutData:
        # if self.type == "Gemm":
        #     return GemmIn2Out
        return ConvPoolIn2Out

    def get_op_type(self) -> OpType:
        if self.type == "Conv":
            return OpType.StdConv
        if self.type == "Gemm":
            return OpType.Gemm
        if self.type == "Depthwise":
            return OpType.Depthwise
        if self.type == "UpsampleBilinear":
            return OpType.UpsampleBilinear
        raise ValueError(f"Type cannot be {self.type}")

    def get_alignment(self) -> Alignment:
        if self.type == "Conv":
            return StdConvAlign
        if self.type == "Gemm":
            return GEMMAlign
        if self.type == "Depthwise" or self.type == "UpsampleBilinear":
            return DWConvAlign
        return NullAlign

    def merge_normalization(self):
        if self.bn is not None:
            channelwise_scale = np.divide(self.bn[0, :], self.bn[3, :])
            channelwise_bias = np.subtract(
                self.bn[1, :],
                np.multiply(self.bn[0, :], np.divide(self.bn[2, :], self.bn[3, :])),
            )
            self.bias = channelwise_bias if channelwise_bias.any() else None
            self.weight = (
                channelwise_scale[:, np.newaxis, np.newaxis, np.newaxis] * self.weight
            )
            # self.bn = None

    def tensor_to_midap_tensor(self):
        if self.order == "NCHW":
            super(ConvOp, self).tensor_to_midap_tensor()
            # Weight -> NCHW -> NHWC
            self.weight = self.weight.transpose(0, 2, 3, 1)
            self.weight_origin = self.weight

    def get_macs(self):  # Overrided
        if self.type == "UpsampleBilinear":
            return self.output_tensor.size * 4
        else:
            return self.output_tensor[:, :, :, 0].size * self.weight.size

    def flip_operation(self, flip):
        if self.reversed ^ flip:
            super(ConvOp, self).flip_operation(flip)
            self.weight = np.flip(self.weight, axis=1)  # NH[WC]

    def __repr__(self):
        return super(ConvOp, self).__repr__() + "kernel shape: {}\n".format(
            self.weight.shape
        )
