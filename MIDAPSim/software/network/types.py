from __future__ import annotations

from enum import Enum, auto

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


class VTensorType(Enum):
    Default = auto()
    InputLinear = auto()
    InputValid = auto()
    OutputLinear = auto()
    OutputReorg = auto()
    OutputWMEM = auto()


class TensorType(Enum):
    In = auto()
    Out = auto()
    Constant = auto()
    Temporal = auto()
    Empty = auto()


class OpType(Enum):
    StdConv = auto()
    Depthwise = auto()
    Gemm = auto()
    MatMul = auto()
    MatMulTrans = auto()
    MaxPool = auto()
    AvgPool = auto()
    GlobalPool = auto()
    Mul = auto()
    Sum = auto()
    WeightedSum = auto()
    Reorg = auto()
    Upsample = auto()
    UpsampleBilinear = auto()
    Concat = auto()
    Crop = auto()
    RoPE = auto()
    Dummy = auto()
    HostProcess = auto()
    Test = auto()

class ActivationType(Enum):
    Linear = auto()
    ReLU = auto()
    Sigmoid = auto()
    LeakyRelu = auto()
    GELU = auto()

    @staticmethod
    def str2act(act_str: str):
        if act_str is None:
            return ActivationType.Linear
        act_str = act_str.lower()
        if act_str in ["linear"]:
            return ActivationType.Linear
        if act_str in ["relu"]:
            return ActivationType.ReLU
        if act_str in ["leakyrelu"]:
            return ActivationType.LeakyRelu
        if act_str in ["sigmoid"]:
            return ActivationType.Sigmoid
        if act_str in ["gelu"]:
            return ActivationType.GELU
        raise ValueError(f"Error: Unknown activation type: {act_str}")
