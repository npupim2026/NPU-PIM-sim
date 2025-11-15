from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from config import cfg
from software.network.quant_info import LayerQuantInfo, QuantType, TensorQuantInfo
from software.network.types import ActivationType, OpType, TensorType

from .tensor import Tensor
from .input_info import InputInfo
from .op_info import OpInfo
from .output_info import OutputInfo

if TYPE_CHECKING:
    from typing import List

    from software.generic_op.operator_base import OperatorBase

    from .op_info import OpParams

    from .virtual_tensor import VirtualTensor

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang", "Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class Layer(object):
    __slots__ = [
        "__name",
        "__op_info",
        "__input_info",
        "__output_info",
        "__quant_info",
        "__parent"
    ]

    def __init__(
        self,
        name: str,
        input_info: InputInfo,
        output_info: OutputInfo,
        op_info: OpInfo,
        quant_info: LayerQuantInfo = None,
    ):
        self.__name = name
        self.__input_info = input_info
        self.__output_info = output_info
        self.__op_info = op_info
        self.__quant_info = quant_info
        self.__parent = None

    @classmethod
    def from_op(cls, op: OperatorBase, input_info: dict) -> Layer:
        __name = op.name
        __input_info = InputInfo(**input_info)
        __output_info = OutputInfo.from_op(op, __input_info.vtensors)
        __op_info = OpInfo.from_op(op, __input_info.is_first_layer)
        __quant_info = None if op.act_scale_32b_to_16b is None and op.act_scale_16b_to_8b is None \
            else LayerQuantInfo(
                type=QuantType.Symm,
                scale_32b_to_16b=op.act_scale_32b_to_16b,
                scale_16b_to_8b=op.act_scale_16b_to_8b
            )
        layer = cls(__name, __input_info, __output_info, __op_info, __quant_info)
        in_layer: Layer
        def setup_hp(layer : Layer):
            layer.op_info.type = OpType.HostProcess
            layer.op.type = OpType.HostProcess
            layer.out_tensor.type = TensorType.In
            for vtensor in layer.in_vtensors:
                vtensor.tensor.type = TensorType.Out
        for in_layer in input_info["layers"]:
            in_layer.__output_info.add_output(layer)
        if layer.op.type == OpType.HostProcess:
            setup_hp(layer)
        if layer.virtual and not cfg.MODEL.ALLOW_ABSTRACT_DATA:
            setup_hp(layer)
        if layer.op.type in [OpType.MaxPool, OpType.AvgPool, OpType.GlobalPool, OpType.Depthwise, OpType.Mul, OpType.Sum, OpType.WeightedSum] and not cfg.MIDAP.WMEM.USE_EXTENSION:
            setup_hp(layer)
        if layer.op.type == OpType.GlobalPool and not cfg.MODEL.REDUCTION_LOGIC:
            setup_hp(layer)
        if not __input_info.layers and cfg.MIDAP.CONTROL_STRATEGY.FIRST_LAYER == "EXCLUDE":
            layer.in_tensor.type = TensorType.Empty
            setup_hp(layer)
        return layer

    def __copy__(self) -> Layer:
        from copy import copy

        name = self.name
        in_info = InputInfo([], [copy(vt) for vt in self.in_vtensors])
        out_info = OutputInfo(vtensor=copy(self.out_vtensor), layers=[])
        op_info = copy(self.op_info)
        quant_info = copy(self.quant_info)
        return Layer(name, in_info, out_info, op_info, quant_info)

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, parent):
        self.__parent = parent

    @property
    def name(self) -> str:
        return self.__name

    @property
    def op_info(self) -> OpInfo:
        return self.__op_info

    @property
    def op(self) -> OpParams:
        return self.__op_info.param

    @property
    def inputs(self) -> List[Layer]:
        return self.__input_info.layers

    @property
    def input(self) -> Layer:
        # XXX We assume that the first layer's output is stored in FMEM.
        return self.__input_info.layers[0] if self.__input_info.layers else None

    @property
    def num_inputs(self) -> int:
        return self.__input_info.num

    @property
    def in_vtensors(self) -> List[VirtualTensor]:
        return self.__input_info.vtensors

    @property
    def in_vtensor(self) -> VirtualTensor:
        return self.__input_info.vtensors[0] if self.in_vtensors else None

    @property
    def in_tensor(self) -> Tensor:
        return self.in_vtensor.tensor if self.in_vtensor is not None else None

    @property
    def outputs(self) -> List[Layer]:
        return self.__output_info.layers

    @property
    def num_outputs(self) -> int:
        return self.__output_info.num

    @property
    def out_vtensor(self) -> VirtualTensor:
        return self.__output_info.vtensor

    @property
    def out_tensor(self) -> Tensor:
        return self.out_vtensor.tensor

    @property
    def is_quantized(self) -> bool:
        return (
            self.__quant_info is not None
            and self.__quant_info.type != QuantType.Default
        )

    @property
    def quant_info(self) -> LayerQuantInfo:
        return self.__quant_info

    @quant_info.setter
    def quant_info(self, quant_info: LayerQuantInfo):
        self.__quant_info = quant_info

    @property
    def virtual(self):
        return self.op.type in [OpType.Upsample, OpType.Concat, OpType.Crop]
    
    @property
    def dummy(self):
        return self.op.type in [OpType.Upsample, OpType.Concat, OpType.Crop, OpType.Dummy, OpType.HostProcess]

    @property
    def test_layer(self):
        return self.op.type == OpType.Test
    
    @property
    def reduction(self): # FIXME: Tag) Reduction
        from config import cfg
        return all([
            cfg.MODEL.REDUCTION_LOGIC,
            self.op.type == OpType.GlobalPool,
            self.inputs and not self.inputs[0].virtual,
            self.out_tensor.shape[-1] <= cfg.MIDAP.REDUCTION.NUM_ENTRIES
        ])

    def __repr__(self) -> str:
        return self.name

    def set_prev(self, layer: Layer):
        self.__input_info.set_prev(layer)

    @property
    def input_scale(self):
        import math

        if not all(
            [self.in_vtensors] +
            [vt.tensor.is_quantized for vt in self.in_vtensors]
            + [t.is_quantized for t in self.op.get_tensors()]
        ):
            return 1.0
        ret = self.in_tensor.quant_info.scale
        if self.op.type in [OpType.Gemm, OpType.StdConv, OpType.Depthwise, OpType.UpsampleBilinear]:
            return ret * self.op.weight.quant_info.scale
        elif self.op.type in [OpType.Mul, OpType.MatMul, OpType.MatMulTrans, OpType.RoPE]:
            return ret * self.in_vtensors[-1].tensor.quant_info.scale
        elif self.op.type in [OpType.AvgPool, OpType.GlobalPool]:
            return ret / (2 ** math.ceil(math.log2(self.op.k_h * self.op.k_w)))
        elif self.op.type in [OpType.WeightedSum]:
            return min(ret, self.in_vtensors[-1].tensor.quant_info.scale)
        else:
            return ret

    @property
    def output_scale(self):
        if not self.out_tensor.is_quantized:
            return 1.0
        return self.out_tensor.quant_info.scale

    def get_tensors(self):
        tensors = self.op.get_tensors()
        if self.is_quantized and self.quant_info.activation_lut is not None:
            tensors.append(self.quant_info.activation_lut)
        return tensors

    def recompute_op(self):
        import numpy as np
        import torch
        import math
        if self.dummy and not self.virtual:
            self.quant_info.scale_32b_to_16b = 1
            self.quant_info.scale_16b_to_8b = self.input_scale / self.output_scale
        if self.dummy:
            return
        input_tensors = [vt.to_torch_tensor() for vt in self.in_vtensors]
        if self.in_vtensors[0].flip_x != self.in_vtensors[-1].flip_x:
            input_tensors[-1] = torch.flip(input_tensors[-1], axis=3)  # NCHW
        if self.is_quantized:
            if self.op.type == OpType.WeightedSum:
                weight_data = np.concatenate([
                    np.full((1, 1, 1, 64), self.in_tensor.quant_info.scale / self.input_scale),
                    np.full((1, 1, 1, 64), self.in_vtensors[-1].tensor.quant_info.scale / self.input_scale)
                ], axis = -2).astype(np.int8)
                self.op.weight = Tensor(self.name + "_w", weight_data, TensorType.Constant, quant_info = TensorQuantInfo(type=QuantType.Symm))
            pad_func, qop, qact = self.op.to_quant_op_sequence()
            input_tensors = [pad_func(t) for t in input_tensors]
            if self.quant_info.scale_32b_to_16b is None:
                #qop.initialize(1.0, self.quant_info.bias_scale)
                #result1 = qop(*input_tensors, n=32).detach().numpy()
                #output_scale = math.ceil(math.log2(np.max(np.abs(result1)) + 1))
                #self.quant_info.scale_32b_to_16b = 2 ** (15 - output_scale) # To 16 bits
                self.quant_info.scale_32b_to_16b = (self.input_scale / self.output_scale) * (2 ** 8)
            qop.initialize(self.quant_info.scale_32b_to_16b, self.quant_info.bias_scale)
            if self.quant_info.scale_16b_to_8b is None:
                #if self.op.activation in [ActivationType.Linear, ActivationType.ReLU]:
                #    self.quant_info.scale_16b_to_8b = (self.input_scale / self.output_scale) / self.quant_info.scale_32b_to_16b
                #else:
                #    self.quant_info.scale_16b_to_8b = 2 ** (-8)
                self.quant_info.scale_16b_to_8b = 2 ** (-8)
            if self.input_scale / self.output_scale != self.quant_info.scale_32b_to_16b * self.quant_info.scale_16b_to_8b:
                self.quant_info.scale_16b_to_8b = self.input_scale / self.output_scale / self.quant_info.scale_32b_to_16b
            qact.scale = self.quant_info.scale_16b_to_8b
            qact.build_lut(
                self.input_scale / self.quant_info.scale_32b_to_16b,
                self.output_scale * self.quant_info.scale_16b_to_8b,
                16, 8
            )
            if self.op.activation not in [ActivationType.Linear, ActivationType.ReLU]:
                self.quant_info.set_lut(
                    self.op.name + "_lut",
                    np.array(qact.lut, dtype=np.int16),
                )

            if self.op.type == OpType.Test:
                return

            result = qact(qop(*input_tensors))
            output_data = (
                result.detach().numpy()[:, :, :, :].transpose(0, 3, 2, 1).astype(np.int8)
            )
            _, ox, oy, oz = self.out_vtensor.shape
            oxo, oyo, ozo = self.out_vtensor.offset
            self.out_tensor.data[
                :, oxo : oxo + ox, oyo : oyo + oy, ozo : ozo + oz
            ] = output_data[:, :ox, :oy, :oz]
        else:
            raise NotImplementedError

    def estimate_overhead(self, *args, **kwargs): # TODO: args, kwargs
        import numpy as np
        from config import cfg
        computation_overhead = 0
        memory_overhead = 0
        if self.__input_info.is_first_layer: # It should be given as input parameter
            memory_overhead += self.in_vtensor.total_size
        if self.op.type in [OpType.Gemm, OpType.StdConv, OpType.Depthwise]:
            computation_overhead += np.prod(self.out_vtensor.shape[1:-1]) * self.op.weight.data.size
            memory_overhead += self.op.weight.data.size
        elif self.op.type in [OpType.Mul, OpType.Sum, OpType.WeightedSum]:
            computation_overhead += self.in_vtensor.total_size
            memory_overhead += sum([vt.total_size for vt in self.in_vtensors])
        elif self.op.type in [OpType.AvgPool, OpType.GlobalPool, OpType.MaxPool]:
            computation_overhead += self.out_vtensor.total_size * self.op.k_h * self.op.k_w
        comp_div =(
            cfg.MIDAP.WMEM.NUM * cfg.MIDAP.SYSTEM_WIDTH
            if self.op.type in [OpType.Gemm, OpType.StdConv]
            else cfg.MIDAP.SYSTEM_WIDTH
        )
        mem_bandwidth = (cfg.SYSTEM.BANDWIDTH * 1000 // cfg.SYSTEM.FREQUENCY) // cfg.SYSTEM.DATA_SIZE
        computation_overhead /= comp_div
        memory_overhead /= mem_bandwidth
        return max(computation_overhead, memory_overhead)
