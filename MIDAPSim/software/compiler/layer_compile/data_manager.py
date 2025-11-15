from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from .feature_data import FeatureData

if TYPE_CHECKING:
    from typing import List

    from .intermediate_info import Status
    from .layer_info import LayerInfo


class DataManager:
    def __init__(self, layer: LayerInfo, bank_size: int, tensor_idx : int = 0):
        self.__status = None
        self.__op = layer.op_info
        self.__layer = layer
        self.__in_tensor = layer.ordered_input_tensors[tensor_idx]
        self.__in_vtensor = layer.ordered_input_vtensors[tensor_idx]
        self.__plane_size = self.in_tensor.yz_plane_size
        self.__num_plane = bank_size // self.__plane_size
        assert self.__in_vtensor.tensor == self.__in_tensor
        

    def set_status(self, status: Status):
        self.__status = status

    @property
    def fmem(self):
        return self.__status.fmem

    @property
    def in_tensor(self):
        # return self.__layer.ordered_input_tensors[0]
        return self.__in_tensor

    @property
    def out_tensor(self):
        return self.__layer.out_tensor

    @property
    def last_load_x(self):
        return self.__status.last_load[self.in_tensor]

    @last_load_x.setter
    def last_load_x(self, x: int):
        self.__status.last_load[self.in_tensor] = x

    @property
    def last_process_x(self):
        return self.__status.last_process[self.in_tensor]

    @last_process_x.setter
    def last_process_x(self, x: int):
        self.__status.last_process[self.in_tensor] = x

    @property
    def is_process_all(self):
        vt = self.__in_vtensor
        return vt.offset[0] + vt.vwidth // vt.scale[0] <= self.last_process_x

    def __get_overlap_width(self, last_x):
        kern_width = self.__op.param.kernel[0]
        stride = self.__op.param.stride
        pad = self.__op.param.pad_l
        start_x = (last_x + pad - kern_width) // stride * stride - pad + stride
        width = last_x - start_x
        return width if width < kern_width else 0

    def __create_data2load(self, last_x: int, overlap: bool):
        t = self.in_tensor
        vt = self.__in_vtensor
        if overlap:
            last_x -= self.__get_overlap_width(last_x)
        width = min(self.__num_plane, vt.offset[0] + vt.vwidth // vt.scale[0] - last_x)
        return FeatureData.initialize(t, last_x, width)

    def __get_num_rest_input(self):
        vt = self.__in_vtensor
        return ceil((vt.offset[0] + vt.vwidth // vt.scale[0] - self.last_load_x) / self.__num_plane)

    def __check_overlap(self):
        return self.__in_vtensor.get_input_x(self.last_load_x) != 0 and self.last_load_x == self.last_process_x

    def get_input_data(self, last_x: int, num: int, is_overlapped: bool = False):
        features: List[FeatureData] = []
        for _ in range(num):
            features.append(self.__create_data2load(last_x, is_overlapped))
            last_x = features[-1].last_x
            is_overlapped = False
        return features

    def get_available2load(self):
        is_overlapped = self.__check_overlap()
        num = min(self.fmem.num_available_banks, self.__get_num_rest_input())
        return self.get_input_data(self.last_load_x, num, is_overlapped)

    def __get_data2process(self, last_x: int):
        fmem = self.fmem
        for d in fmem:
            if d.tensor == self.in_tensor and last_x in range(d.pivot, d.last_x):
                return d
        return None

    def __get_num_input_in_fmem(self):
        t = self.in_tensor
        fmem = self.fmem
        num_input = 0
        for d in fmem:
            if d.tensor == t:
                num_input += 1
        return num_input

    def get_available2process(self, num: int):
        last_x = self.last_process_x
        num = min(num, self.__get_num_input_in_fmem())
        features: List[FeatureData] = []
        for _ in range(num):
            data = self.__get_data2process(last_x)
            if not data:
                break
            if data not in features:
                features.append(data)
                last_x = features[-1].last_x
        return features

    def get_outbank_from_input(self, in_data: List[FeatureData], num: int):
        return self.__op.in2out.input2output(self.__layer, in_data, num)
