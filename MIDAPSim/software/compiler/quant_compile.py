from __future__ import annotations

import logging
from software.network.model import ModelGraph
from software.network.quant_info import LayerQuantInfo, QuantType, TensorQuantInfo

import numpy as np
import math

from software.network.types import OpType, TensorType
from software.network.tensor import Tensor
from typing import TYPE_CHECKING

from logger import init_logger

from . import CompileTechnique

if TYPE_CHECKING:
    from software.compiler.compile_info import CompileInfo
    from software.network.model import Layer
    from typing import Dict


__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("Quantize", logging.INFO)


class QuantCompile(
    CompileTechnique
):  # TODO: Quantization must be finished before the model is given as an input
    @classmethod
    def compile(cls, info: CompileInfo):
        model = info.model
        dep_dict = dict()
        model.traverse(lambda l: cls.__get_dependency_dict(l, dep_dict))
        for t in model.tensors:
            cls.__quantize_tensor(t, dep_dict)
        model.traverse(lambda l: cls.__quantize_layer(model, l), topological=True)
        logger.info("Quantization Done")
        return info

    @staticmethod
    def __get_dependency_dict(layer: Layer, dependency_dict: Dict[Tensor, Tensor]): # FIXME
        if layer.op.type in [OpType.Sum]:
            dependency_dict[layer.in_vtensors[0].tensor] = layer.in_vtensors[1].tensor
            dependency_dict[layer.in_vtensors[1].tensor] = layer.in_vtensors[0].tensor

    @staticmethod
    def __quantize_tensor(tensor: Tensor, dependency_dict: Dict[Tensor, Tensor]):
        # Constants are quantized in quantize_layer step
        epsilon = 0 if tensor.type == TensorType.Constant else 0.1
        scale = 1.0
        quant_info = tensor.quant_info
        if tensor in dependency_dict:
            pair = dependency_dict[tensor]
            if pair.is_quantized:
                if tensor.is_quantized:
                    quant_info = tensor.quant_info
                    tensor.quant_info.scale = max(tensor.quant_info.scale, pair.quant_info.scale)
                else:
                    quant_info = pair.quant_info
            elif tensor.is_quantized:
                quant_info = tensor.quant_info
            else:
                pair_epsilon = 0 if pair.type == TensorType.Constant else 0.1
                tensor_max = np.max(np.abs(tensor.data)) + epsilon
                pair_max = np.max(np.abs(pair.data)) + pair_epsilon
                scale = 2 ** math.ceil(
                    math.log2(max(tensor_max, pair_max) / 128.0)
                )
                quant_info = TensorQuantInfo(type=QuantType.Symm, scale=scale, bits=8)
        if quant_info is None:
            scale = 2 ** math.ceil(
                math.log2((np.max(np.abs(tensor.data)) + epsilon) / 128.0)
            )
            quant_info = TensorQuantInfo(type=QuantType.Symm, scale=scale, bits=8)
        else:
            scale = quant_info.scale
        tensor.quant_info = quant_info
        if tensor.type in [TensorType.In, TensorType.Constant]:
            # logger.debug(f"Tensor {tensor.name}: Maximum value = {np.max(np.abs(tensor.data))}, Scale becomes {scale}")
            data = tensor.data / scale
            tensor.data = data.astype(np.int8)
        else:
            tensor.data = np.zeros(tensor.shape, dtype=np.int8)

    @staticmethod
    def __quantize_layer(model: ModelGraph, layer: Layer): # Temporal Implementation
        # if layer.virtual:
        #     return
        logger.debug(f"Quantize layer {layer}")
        bias_scale = 1.0
        input_scale = layer.input_scale
        bias = layer.op.bias

        if bias is not None:
            bs = bias.quant_info.scale / input_scale
            if bs > 1:
                bias_scale = bs
            else:
                bias.data = np.multiply(bias.data, bs).astype(np.int8)
            logger.debug(f"bias scale = {bias_scale}")
        
        if layer.quant_info is None:
            layer.quant_info = LayerQuantInfo(type=QuantType.Symm)
        layer.quant_info.bias_scale = bias_scale
        layer.recompute_op()
        model.register_items(layer)
