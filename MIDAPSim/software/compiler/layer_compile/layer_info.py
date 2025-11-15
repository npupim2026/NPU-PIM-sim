from __future__ import annotations

import logging
from functools import reduce
from typing import TYPE_CHECKING, Set

import attr
from config import cfg
from logger import init_logger
from software.network.model import ModelGraph
from software.network.types import OpType

from .behavior import Action, SystemBehavior, Behavior
from .mapping import MappingInfo

if TYPE_CHECKING:
    from typing import List

    from software.network import Layer
    from software.network.op_info import OpInfo, OpParams
    from software.network.tensor import Tensor
    from software.network.virtual_tensor import VirtualTensor

    from .feature_data import FeatureData, FeatureFragment


__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


logger = init_logger("Layer Compile", logging.INFO)


@attr.s(slots=True)
class StationaryInfo(object):
    # Stationary Value
    # -1 : not determined
    # 0 ~ # of fmem banks
    input: int = attr.ib(default=-1)
    output: int = attr.ib(default=-1)


@attr.s(slots=True)
class LayerInfo(object):
    layer: Layer = attr.ib()
    op_info: OpInfo = attr.ib()
    stationary: StationaryInfo = attr.ib(default=attr.Factory(StationaryInfo))
    mapping: MappingInfo = attr.ib(default=attr.Factory(MappingInfo))
    behavior: List[Behavior | SystemBehavior] = attr.ib(default=attr.Factory(list))
    reversed: bool = attr.ib(default=True)

    @classmethod
    def from_layer(cls, layer: Layer):
        if layer.op.type in [OpType.Concat, OpType.Upsample, OpType.Crop, OpType.Dummy, OpType.HostProcess]:
            return cls(layer, layer.op_info, reversed=False)
        reverse_write = cls.__determine_reverse_write(layer)
        return cls(layer, layer.op_info, reversed=reverse_write)

    @staticmethod
    def __determine_reverse_write(layer: Layer):
        model: ModelGraph = layer.parent
        outputs: Set[Layer] = model.get_next_node(layer)
        need_ordered_out = any(
            [layer.outputs == [], layer.out_tensor.shared] +
            [l.op.type in [OpType.MatMul, OpType.MatMulTrans, OpType.RoPE] for l in outputs] +
            [l.op.type == OpType.Concat for l in outputs] +
            [l.op.type == OpType.WeightedSum for l in outputs] +
            [l.op.weight is not None and l.op.weight.shared for l in outputs]
        )
        reverse_write = not need_ordered_out or layer.in_tensor.flip_x
        layer.out_tensor.flip_x = reverse_write ^ layer.in_tensor.flip_x
        layer.op.flip_x = layer.in_tensor.flip_x
        return reverse_write

    def load_ops(self) -> List[Behavior]:
        return [b for b in self.behavior if isinstance(b, Behavior) and b.is_load()]

    def input2output(self, inputs: FeatureData) -> FeatureFragment:
        return self.op_info.in2out.input2output(self, inputs, cfg.MIDAP.FMEM.NUM)

    @property
    def name(self) -> str:
        return self.layer.name

    @property
    def op(self) -> OpParams:
        return self.layer.op

    @property
    def in_vtensor(self) -> VirtualTensor:
        return self.layer.in_vtensor

    @property
    def out_vtensor(self) -> VirtualTensor:
        return self.layer.out_vtensor

    @property
    def in_tensor(self) -> Tensor:
        return self.layer.in_tensor

    @property
    def out_tensor(self) -> Tensor:
        return self.layer.out_tensor

    def num_load_bank(self, tensor: Tensor = None) -> int:
        load_list = self.load_ops()
        if tensor is None:
            return reduce(lambda x, y: x + y.num_banks, [0] + load_list)
        return reduce(lambda x, y: x + len(list(filter(lambda data: data.tensor == tensor, y.data))), [0] + load_list)

    def num_input_mapping(self, tensor: Tensor = None) -> int:
        if tensor is None:
            return len(self.mapping.input)
        return len(list(filter(lambda m: m.data.tensor == tensor, self.mapping.input)))

    def num_init_input(self, tensor: Tensor = None) -> int:
        return self.num_input_mapping(tensor) - self.num_load_bank(tensor)

    def load_access_size(self) -> int:
        return sum(
            [
                sum([d.size for d in b.data])
                for b in self.behavior
                if b.action == Action.LOAD
            ]
        )

    def store_access_size(self) -> int:
        save_on_fmem = sum([m.data.size for m in self.mapping.output])
        return self.layer.out_vtensor.total_size - save_on_fmem

    def is_weight_larger_than_wmem(self):
        op_info = self.op_info
        if op_info.param.weight is None or op_info.type == OpType.Depthwise:
            return False

        wmem_size = cfg.MIDAP.WMEM.NUM_ENTRIES * cfg.MIDAP.WMEM.NUM
        multi_bank_process = op_info.param.weight.total_size > wmem_size
        return multi_bank_process

    def weight_load_size(self) -> int:
        num_weight_load = 1
        if self.is_weight_larger_than_wmem():
            num_weight_load = reduce(
                lambda x, y: x + (1 if y.action == Action.PROCESS else 0),
                [0] + self.behavior,
            )
        weight_size = 0
        op_param = self.op_info.param
        if op_param.weight is not None:
            weight_size = op_param.weight.total_size * num_weight_load
        return weight_size

    @property
    def virtual(self):
        return self.layer.virtual

    @property
    def dummy(self):
        return self.layer.dummy

    @property
    def reduction(self): #FIXME: Tag) Reduction
        return self.layer.reduction            

    @property
    def ordered_input_tensors(self) -> List[Tensor]:
        if self.num_input_mapping() == 0:
            return [v.tensor for v in self.layer.in_vtensors]
        else:
            main_input = self.mapping.input[0].data.tensor
            input_tensors = [main_input]
            for vtensor in self.layer.in_vtensors:
                if vtensor.tensor not in input_tensors:
                    input_tensors.append(vtensor.tensor)
            return input_tensors

    @property
    def ordered_input_vtensors(self) -> List[VirtualTensor]:
        if self.num_input_mapping() == 0:
            return self.layer.in_vtensors
        else:
            main_input = self.mapping.input[0].data.tensor
            input_vtensors = []
            for vtensor in self.layer.in_vtensors:
                if vtensor.tensor != main_input:
                    input_vtensors.append(vtensor)
                else:
                    input_vtensors = [vtensor] + input_vtensors
            return input_vtensors
