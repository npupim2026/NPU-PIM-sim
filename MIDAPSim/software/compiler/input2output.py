from __future__ import annotations

from abc import ABC, abstractclassmethod
from typing import TYPE_CHECKING
from math import ceil

from data_structure.virtual_tensor import VirtualTensor

from .layer_compile.feature_data import FeatureData, FeatureFragment

if TYPE_CHECKING:
    from typing import List

    from network.op_info import OpParams

    from .layer_compile import LayerInfo

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class InData2OutData(ABC):
    @abstractclassmethod
    def input2output(
        cls, info: LayerInfo, in_data: List[FeatureData], num: int
    ) -> FeatureFragment:
        pass

    @classmethod
    def generate_feature_frags(
        cls,
        info: LayerInfo,
        out_vtensor: VirtualTensor,
        num: int,
        offset: int,
        last_x: int,
    ):
        outputs = FeatureFragment.initialize(out_vtensor.tensor, reverse=info.reversed)
        outputs = reversed(outputs[:num]) if info.reversed else outputs[:num]
        return FeatureFragment(filter(lambda d: d.overlap(offset, last_x), outputs))


class VirtualIn2Out(InData2OutData):
    @classmethod
    def input2output(
        cls, info: LayerInfo, in_data: List[FeatureData], num: int
    ) -> FeatureFragment:
        raise NotImplementedError("This in2out must not be called")


class DefaultIn2Out(InData2OutData):
    @classmethod
    def input2output(
        cls, info: LayerInfo, in_data: List[FeatureData], num: int
    ) -> FeatureFragment:
        in_vtensor = info.in_vtensor
        out_vtensor = info.out_vtensor
        offset = out_vtensor.get_output_x(in_vtensor.get_input_x(in_data[0].pivot))
        last_x = out_vtensor.get_output_x(in_vtensor.get_input_x(in_data[-1].last_x))
        return cls.generate_feature_frags(info, out_vtensor, num, offset, last_x)


class GemmIn2Out(InData2OutData):
    @classmethod
    def input2output(
        cls, info: LayerInfo, in_data: List[FeatureData], num: int
    ) -> FeatureFragment:
        out_vtensor = info.layer.out_vtensor
        offset = in_data[0].pivot // out_vtensor.height
        last_x = in_data[-1].last_x // out_vtensor.height
        return cls.generate_feature_frags(info, out_vtensor, num, offset, last_x)


class ConvPoolIn2Out(InData2OutData):
    @staticmethod
    def _translate_x(in_x: int, op: OpParams) -> int:
        x = 0 if in_x == 0 else in_x + op.pad_l
        return ceil(x / op.stride)

    @classmethod
    def input2output(
        cls, info: LayerInfo, in_data: List[FeatureData], num: int
    ) -> FeatureFragment:
        op: OpParams = info.op
        in_vtensor = info.in_vtensor
        out_vtensor = info.out_vtensor
        offset = cls._translate_x(in_vtensor.get_input_x(in_data[0].pivot), op)
        offset = out_vtensor.get_output_x(offset)
        last_x = cls._translate_x(in_vtensor.get_input_x(in_data[-1].last_x), op)
        last_x = out_vtensor.get_output_x(last_x)
        last_x = min(out_vtensor.width, last_x)
        # print(f"{info.layer}: [{in_data[0].pivot}, {in_data[-1].last_x}] -> [{offset}, {last_x}]")
        return cls.generate_feature_frags(info, out_vtensor, num, offset, last_x)


class GlobalPoolIn2Out(InData2OutData):
    @classmethod
    def input2output(
        cls, info: LayerInfo, in_data: List[FeatureData], num: int
    ) -> FeatureFragment:
        return FeatureFragment.initialize(info.out_tensor, reverse=info.reversed)
