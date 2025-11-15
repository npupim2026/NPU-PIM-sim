from __future__ import annotations

import logging
from software.compiler.wmem_info import ComputeType
from typing import TYPE_CHECKING

import attr
from logger import init_logger

from .types import ActivationType, OpType, VTensorType

if TYPE_CHECKING:
    from typing import List, Tuple

    from software.compiler.align_compile import Alignment
    from software.compiler.input2output import InData2OutData
    from software.generic_op.operator_base import OperatorBase

    from .tensor import Tensor

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang" "Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"

logger = init_logger("Op", logging.INFO)


@attr.s(slots=True, init=False)
class OpParams(object):
    name: str = attr.ib()
    activation: ActivationType = attr.ib(validator=attr.validators.in_(ActivationType))
    type: OpType = attr.ib(validator=attr.validators.in_(OpType))
    kernel: Tuple(int, int) = attr.ib()
    pad: Tuple(int, int, int, int) = attr.ib()
    stride: int = attr.ib()
    dilation: int = attr.ib()
    broadcast: bool = attr.ib()
    in_plane: bool = attr.ib()
    weight: Tensor = attr.ib()
    bias: Tensor = attr.ib()
    macs: int = attr.ib()
    test_code: list = attr.ib()
    mapping: str = attr.ib()
    __flip_x: bool = attr.ib()

    def __init__(self, op: OperatorBase):
        self.name = op.name
        self.activation = ActivationType.str2act(op.activation)
        self.type = op.get_op_type()
        self.kernel = op.kernel  # k_w, k_h
        self.pad = op.pad  # top, bottom, left, right
        self.stride = op.stride
        self.dilation = op.dilation
        self.broadcast = op.broadcast
        self.in_plane = op.in_plane
        self.weight = op.get_weight_tensor()
        self.bias = op.get_bias_tensor()
        self.macs = op.get_macs()
        self.test_code = op.test_code
        self.mapping = op.mapping if op.get_op_type() == OpType.HostProcess else 'midap'
        self.__flip_x = False

    @property
    def k_w(self):
        return self.kernel[1]

    @property
    def k_h(self):
        return self.kernel[0]

    @property
    def pad_t(self):
        return self.pad[0]

    @property
    def pad_b(self):
        return self.pad[1]

    @property
    def pad_l(self):
        return self.pad[2]

    @property
    def pad_r(self):
        return self.pad[3]

    @property
    def flip_x(self):
        return self.__flip_x

    @flip_x.setter
    def flip_x(self, flip_x: bool = True):
        if flip_x == self.flip_x:
            return
        if self.weight:
            self.weight.flip_x = flip_x
        self.pad = (self.pad[0], self.pad[1], self.pad[3], self.pad[2])
        self.__flip_x = flip_x

    @property
    def is_conv(self):
        return self.type in [OpType.StdConv, OpType.Depthwise, OpType.Gemm, OpType.UpsampleBilinear, OpType.MatMul, OpType.MatMulTrans, OpType.RoPE]

    @property
    def is_elwise(self):
        return self.type in [OpType.Mul, OpType.WeightedSum, OpType.Sum]

    @property
    def is_pool(self):
        return self.type in [OpType.AvgPool, OpType.MaxPool, OpType.GlobalPool]

    @property
    def is_test(self):
        return self.type == OpType.Test

    @property
    def is_dummy(self):
        return not any([self.is_conv, self.is_elwise, self.is_pool, self.is_test])

    @property
    def compute_type(self):
        if self.type == OpType.StdConv:
            return ComputeType.StdConv
        if self.type == OpType.Depthwise:
            return ComputeType.DWConv
        if self.type == OpType.WeightedSum: # FIXME: Temporal
            if self.weight is None:
                return ComputeType.Pool
            else:
                return ComputeType.WeightedSum
        if self.is_elwise:
            return ComputeType.Elwise
        if self.is_pool:
            return ComputeType.Pool
        if self.type == OpType.UpsampleBilinear:
            return ComputeType.WeightedSum
        if self.type == OpType.MatMul:
            return ComputeType.MatMul
        if self.type == OpType.MatMulTrans:
            return ComputeType.MatMulTrans
        if self.type == OpType.RoPE:
            return ComputeType.MatMul
        if self.type == OpType.Test:
            return ComputeType.StdConv  # FIXME: Temporal solution
        return None

    def get_tensors(self) -> List[Tensor]:
        ret = []
        if self.weight is not None:
            ret.append(self.weight)
        if self.bias is not None:
            ret.append(self.bias)
        return ret

    def to_quant_op_sequence(self):
        import numpy as np
        import torch
        from quant_op.mq import QuantOps
        from torch import nn
        op = self
        pad_func = (
            nn.ZeroPad2d((op.pad_l, op.pad_r, op.pad_t, op.pad_b))
            if not self.in_plane
            else nn.ReplicationPad2d((op.pad_l, op.pad_r, op.pad_t, op.pad_b))
        )
        qop = None
        qact = nn.Identity()
        if op.type in [OpType.Gemm, OpType.StdConv, OpType.Depthwise]:
            weight = op.weight.data.astype(np.float32)
            group = 1
            if op.type == OpType.Depthwise:
                if weight.shape[-1] != 1:
                    weight = weight.reshape(1, op.k_w, op.k_h, -1)
                    weight = weight.transpose(3, 1, 2, 0)
                group = weight.shape[0]
            weight = weight.transpose(0, 3, 2, 1)
            in_channel = weight.shape[1] * group
            out_channel = weight.shape[0]
            qop = QuantOps.Conv2d(
                in_channel,
                out_channel,
                [op.k_h, op.k_w],
                op.stride,
                0,
                op.dilation,
                group,
                not self.bias is None
            )
            qop.weight = nn.Parameter(torch.from_numpy(weight))
        elif op.type in [OpType.AvgPool, OpType.GlobalPool]:
            qop = QuantOps.AvgPool2d([op.k_h, op.k_w], op.stride, 0)
        elif op.type in [OpType.MaxPool]:
            qop = QuantOps.MaxPool2d([op.k_h, op.k_w], op.stride, 0)
        elif op.type in [OpType.MatMulTrans]:
            qop = QuantOps.MatMulTrans()
        elif op.type in [OpType.Sum]:
            qop = QuantOps.Add()
        elif op.type in [OpType.WeightedSum]:
            weight = op.weight.data.astype(np.float32)
            qop = QuantOps.WeightedSum((weight[0, 0, 0, 0], weight[0, 0, 1, 0]))
        elif op.type in [OpType.Mul]:
            qop = QuantOps.Mul()
        elif op.type in [OpType.UpsampleBilinear]:
            qop = QuantOps.UpsampleBilinear((op.k_h + 1) // 2)
        elif op.type in [OpType.RoPE]:
            qop = QuantOps.RoPE()
        elif op.type in [OpType.Test]:
            qop = QuantOps.MaxPool2d([1, 1], 1, 0)  # FIXME: Temporal solution
        else:
            raise NotImplementedError(op)
        if op.bias is not None:
            bias = op.bias.data.astype(np.float32).reshape(1, -1, 1, 1)
            qop.bias = nn.Parameter(torch.from_numpy(bias))
        if op.activation is not None:
            if op.activation in [ActivationType.ReLU]:
                qact = QuantOps.ReLU()
            elif op.activation == ActivationType.Linear:
                qact = QuantOps.Identity()
            elif op.activation == ActivationType.Sigmoid:
                qact = QuantOps.Sigmoid()
            elif op.activation == ActivationType.LeakyRelu:
                qact = QuantOps.LeakyReLU()
            elif op.activation == ActivationType.GELU:
                qact = QuantOps.GELU()
            else:
                raise NotImplementedError(op.activation)
        return (pad_func, qop, qact)


@attr.s(slots=True, init=False)
class OpInfo(object):
    type: OpType = attr.ib(validator=attr.validators.in_(OpType))
    param: OpParams = attr.ib()
    alignment: Alignment = attr.ib()
    in2out: InData2OutData = attr.ib()

    def __init__(
        self,
        type: OpType,
        op_param: OpParams,
        alignment: Alignment,
        in2out: InData2OutData,
    ):
        self.type = type
        self.param = op_param
        self.alignment = alignment
        self.in2out = in2out

    @classmethod
    def from_op(cls, op: OperatorBase, is_first_layer: bool = False):
        from config import cfg
        if all([
            op.type == "Conv" and op.weight.shape[-1] < 8,
            is_first_layer,
            cfg.MIDAP.CONTROL_STRATEGY.FIRST_LAYER == 'GEMM'
        ]):
            op.type = "Gemm"
        type = op.get_op_type()
        op_param = OpParams(op)
        alignment = op.get_alignment()
        in2out = op.get_in2out()
        return cls(type, op_param, alignment, in2out)

    def __copy__(self):
        import copy

        op = copy.copy(self.param)
        return OpInfo(self.type, op, self.alignment, self.in2out)
