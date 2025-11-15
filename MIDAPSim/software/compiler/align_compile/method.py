from __future__ import annotations

import logging
from abc import ABC, abstractclassmethod
from software.network.types import TensorType, VTensorType, OpType
from typing import TYPE_CHECKING
from copy import copy

import numpy as np
from config import cfg
from logger import init_logger

from software.network.tensor import Tensor
from software.network.virtual_tensor import VirtualTensor

if TYPE_CHECKING:
    from numpy import typing as npt
    from software.network.layer import Layer

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang", "Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"

logger = init_logger("Compile Operation", logging.INFO)


class Alignment(ABC):
    _num_cims: int = cfg.MIDAP.WMEM.NUM
    _sys_width: int = cfg.MIDAP.SYSTEM_WIDTH
    _addr_align: int = cfg.DRAM.ADDRESS_ALIGN

    @classmethod
    def align(cls, layer: Layer):
        cls._input_weight_align(layer)
        cls._cim_align(layer)
        cls._adder_tree_align(layer)

    @staticmethod
    def _align(array: npt.ArrayLike, align_unit: int, align_axis: int) -> npt.ArrayLike:
        dim = len(array.shape)
        assert align_axis < dim and 0 <= align_axis
        pad: int = array.shape[align_axis] % align_unit
        if pad > 0:
            pad = align_unit - pad
            paddings = tuple(
                (0, 0) if i != align_axis else (0, pad) for i in range(dim)
            )
            array = np.pad(array, paddings, "constant")
        return array

    @staticmethod
    def _tensor_align(tensor: Tensor, align_unit: int, align_axis: int):
        tensor.data = Alignment._align(tensor.data, align_unit, align_axis)

    @staticmethod
    def _vtensor_align(vtensor: VirtualTensor, align_unit: int, align_axis: int):
        dim = len(vtensor.shape)
        assert align_axis < dim and 0 <= align_axis
        pad: int = vtensor.shape[align_axis] % align_unit
        pad_align = (align_unit - pad) % align_unit
        if pad_align == 0:
            pass
        elif any(
            [vtensor.scale[align_axis - 1] != 1, vtensor.offset[align_axis - 1] != 0]
        ):
            raise RuntimeError(
                f"Cannot align to virtualized tensor {vtensor}, axis = {align_axis}"
            )
        new_shape = tuple(
            vtensor.shape[i] if i != align_axis else (vtensor.shape[i] + pad_align)
            for i in range(dim)
        )
        vtensor.shape = new_shape
        Alignment._tensor_align(vtensor.tensor, align_unit, align_axis)

    @abstractclassmethod
    def _input_weight_align(cls, layer: Layer):
        pass

    @abstractclassmethod
    def _cim_align(cls, layer: Layer):
        pass

    @abstractclassmethod
    def _adder_tree_align(cls, layer: Layer):
        pass


class StdConvAlign(Alignment):
    @classmethod
    def _input_weight_align(cls, layer: Layer):
        assert len(layer.in_vtensors) == 1
        # TODO
        # Need to consider input virtualization.
        # If input virtualization, align_unit is _sys_width.
        align_unit = (
            cls._num_cims
            if layer.op.dilation == 1 and layer.in_vtensor.type == VTensorType.Default
            else cls._sys_width
        )
        # in_tensor = layer.in_tensors[0]
        # in_tensor.data = cls._align(in_tensor.data, align_unit, 3)
        init_z_shape = layer.in_tensor.shape[-1]
        cls._vtensor_align(layer.in_vtensor, align_unit, 3)
        # TODO Sync weight shape with input tensor. Which case?
        op = layer.op
        # op.weight = cls._align(op.weight, align_unit, 3)  # NWHC
        if op.weight.shape[-1] != init_z_shape:
            align_unit = cls._sys_width
        cls._tensor_align(op.weight, align_unit, 3)


    @classmethod
    def _cim_align(cls, layer: Layer):
        out_tensor = layer.out_vtensor
        align = cls._num_cims
        if out_tensor.shape[-1] >= 128:  ### FIXME: TMEM-size aware padding
            align = cls._sys_width
        # out_tensor.data = cls._align(out_tensor.data, align, 3)  # NWHC
        cls._vtensor_align(out_tensor, align, 3)
        op = layer.op
        # op.weight = cls._align(op.weight, align, 0)  # NWHC
        # op.bias = cls._align(op.bias, align, 0)
        cls._tensor_align(op.weight, align, 0)
        # FIXME: temporal solution for bias padding
        if op.bias is not None:
            cls._tensor_align(op.bias, cls._sys_width, 0)
            logger.debug(f"{op.bias}")

    @classmethod
    def _adder_tree_align(cls, layer: Layer):
        weight = layer.op.weight
        # weight = weight.reshape(*weight.shape[:2], -1)
        weight.data = weight.data.reshape(*weight.shape[:2], -1)
        # layer.op.weight = cls._align(weight, cls._sys_width, 2)  # NW(HC)
        cls._tensor_align(weight, cls._sys_width, 2)


class GEMMAlign(StdConvAlign):
    @classmethod
    def _input_weight_align(cls, layer: Layer):
        pass

    @classmethod
    def _cim_align(cls, layer: Layer):
        pass
        # out_tensor = layer.out_vtensor
        # # out_tensor.data = cls._align(out_tensor.data, cls._num_cims, 3)  # WHC
        # cls._vtensor_align(out_tensor, cls._num_cims, 3)
        # op = layer.op
        # # op.weight = cls._align(op.weight, cls._num_cims, 0)  # NWHC
        # cls._tensor_align(op.weight, cls._num_cims, 0)
        # if op.bias is not None:
        #     cls._tensor_align(op.bias, cls._num_cims, 0)

    @classmethod
    def _adder_tree_align(cls, layer: Layer):
        from utils.func import im2col

        assert len(layer.in_vtensors) == 1
        in_tensor = layer.in_vtensors[0]
        data = im2col(in_tensor.data, layer.op)
        data = data.transpose(0, 3, 2, 1)
        data = data.reshape(
            1, layer.out_vtensor.shape[1], layer.out_vtensor.shape[2], -1
        )
        in_tensor.data = data
        # in_tensor.data = cls._align(data, cls._sys_width, 2)  # WHC

        weight = layer.op.weight.data.transpose(0, 3, 1, 2)  # NWHC -> N(CWH)
        weight = weight.reshape(weight.shape[0], 1, 1, -1)
        layer.op.weight.data = weight
        # layer.op.weight = cls._align(weight, cls._sys_width, 3)  # NW(HC)
        layer.op.kernel = [1, 1]
        layer.op.pad = [0, 0, 0, 0]
        layer.op.stride = 1
        layer.op.dilation = 1
        layer.op.type = OpType.StdConv
        super()._input_weight_align(layer)
        super()._cim_align(layer)
        super()._adder_tree_align(layer)


class DWConvAlign(Alignment):
    @classmethod
    def _input_weight_align(cls, layer: Layer):
        assert len(layer.in_vtensors) == 1
        in_tensor = layer.in_vtensors[0]
        # in_tensor.data = cls._align(in_tensor.data, cls._sys_width, 3)
        cls._vtensor_align(in_tensor, cls._sys_width, 3)

    @classmethod
    def _cim_align(cls, layer: Layer):
        pass

    @classmethod
    def _adder_tree_align(cls, layer: Layer):
        # TODO Sync weight shape with input tensor. Which case?
        op = layer.op
        # op.weight = cls._align(op.weight, cls._sys_width, 0)  # NWHC
        cls._tensor_align(op.weight, cls._sys_width, 0)
        if op.bias is not None:
            cls._tensor_align(op.bias, cls._sys_width, 0)

        out_tensor = layer.out_vtensor
        # out_tensor.data = cls._align(out_tensor.data, cls._sys_width, 3)  # NWHC
        cls._vtensor_align(out_tensor, cls._sys_width, 3)
        weight = op.weight.data
        # weight = weight.transpose(3, 1, 2, 0)
        weight = weight.reshape(weight.shape[0] // cls._sys_width, cls._sys_width, weight.shape[1], weight.shape[2])
        weight = weight.transpose(0, 2, 3, 1)
        weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        op.weight.data = weight  # N x W x H x 1 --> 1 x W x H x N
        # layer.op.weight = weight.reshape(weight.shape[0], weight.shape[1], -1)


class ArithAlign(Alignment):
    align_unit = cfg.MIDAP.SYSTEM_WIDTH

    @classmethod
    def _input_weight_align(cls, layer: Layer):
        pass

    @classmethod
    def _cim_align(cls, layer: Layer):
        pass

    @classmethod
    def _adder_tree_align(cls, layer: Layer):
        for in_tensor in layer.in_vtensors:
            # in_tensor.data = cls._align(in_tensor.data, cls._sys_width, 3)  # NWHC
            cls._vtensor_align(in_tensor, cls._sys_width, 3)

        out_tensor = layer.out_vtensor
        # out_tensor.data = cls._align(out_tensor.data, cls._sys_width, 3)  # WHC
        cls._vtensor_align(out_tensor, cls._sys_width, 3)


class OutAlign(Alignment):
    @classmethod
    def _input_weight_align(cls, layer: Layer):
        pass

    @classmethod
    def _cim_align(cls, layer: Layer):
        pass

    @classmethod
    def _adder_tree_align(cls, layer: Layer):
        out_tensor = layer.out_vtensor
        cls._vtensor_align(out_tensor, cls._sys_width, 3)


class MatMulAlign(Alignment):
    @classmethod
    def _input_weight_align(cls, layer: Layer):
        assert len(layer.in_vtensors) == 2
        align_unit = (
            cls._num_cims
            if layer.in_vtensors[0].type == VTensorType.Default
            else cls._sys_width
        )
        cls._vtensor_align(layer.in_vtensors[0], align_unit, 3)
        cls._vtensor_align(layer.in_vtensors[1], cls._sys_width, 3) # weight


    @classmethod
    def _cim_align(cls, layer: Layer):
        out_tensor = layer.out_vtensor
        align = cls._num_cims
        if out_tensor.shape[-1] >= 128:  ### FIXME: TMEM-size aware padding
            align = cls._sys_width
        cls._vtensor_align(out_tensor, align, 3)
        # Input virtual tensor align for the second input
        prev_layer = None
        for l in layer.inputs:
            if l.out_tensor == layer.in_vtensors[-1].tensor:
                prev_layer = l
        layer.in_vtensors[-1] = copy(layer.in_vtensors[-1])
        if prev_layer is not None and prev_layer.out_vtensor.type != VTensorType.OutputWMEM:
            prev_layer.out_vtensor.type = VTensorType.InputLinear
            if prev_layer.out_tensor.type == TensorType.Temporal:
                new_out_tensor = copy(prev_layer.out_vtensor)
                new_out_tensor.type = VTensorType.OutputWMEM
                prev_layer._Layer__output_info._OutputInfo__vtensor = new_out_tensor
        cls._vtensor_align(layer.in_vtensors[-1], align, 1)
        if layer.op.bias is not None:
            cls._tensor_align(layer.op.bias, cls._sys_width, 0)
            logger.debug(f"{layer.op.bias}")

    @classmethod
    def _adder_tree_align(cls, layer: Layer):
        pass


class NullAlign(Alignment):
    pass
