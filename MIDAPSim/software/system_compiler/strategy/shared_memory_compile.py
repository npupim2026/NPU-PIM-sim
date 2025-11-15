from __future__ import annotations

import logging
from collections import OrderedDict
from software.compiler.wmem_info import ComputeType
from typing import TYPE_CHECKING

from config import cfg
from logger import init_logger
from software.compiler.align_compile import Alignment
from software.network.tensor import Tensor
from software.network.types import TensorType
from software.system_compiler.memory_info import MemoryType

from .strategy import Strategy

if TYPE_CHECKING:
    from software.system_compiler.system_info import SystemInfo


__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("Shared Memory Compile", logging.INFO)


class SharedMemoryCompile(Strategy):
    def _print_status(self, curr):
        def _print_func(l):
            return [str(x) for x in l]

        logger.debug("====================================================")
        logger.debug(f"{_print_func(curr)}\n")
        logger.debug("====================================================")

    def _initialize(self, info: SystemInfo) -> None:
        self._shared_info = info.shared_compile_info
        self._model = info.shared_compile_info.model
        self._dep_info = info.dependency_info
        self._core_info = info.core_info
        self._memory_info = info.memory_info

    def _preprocess(self) -> None:
        # Define shared items
        pass

    def compile(self, info: SystemInfo) -> SystemInfo:
        self._initialize(info)
        self._preprocess()
        self._run()
        self._postprocess()
        info.shared_compile_info.model = self._model
        info.dependency_info = self._dep_info
        info.memory_info = self._memory_info
        return info

    def _run(self) -> None:
        pass

    def _postprocess(self) -> None:
        pass


class Prototype(SharedMemoryCompile):
    def _preprocess(self) -> None:
        self._constants: OrderedDict[str, Tensor] = OrderedDict()
        self._shared_features: OrderedDict[str, Tensor] = OrderedDict()
        self._rearrange_dict: OrderedDict[str, int] = OrderedDict()
        tensors = self._model.tensors
        for name, wmem_info in self._shared_info.wmem_info_dict.items():
            layer = self._shared_info.model._str2layer[name]
            if wmem_info.compute_type in [ComputeType.DWConv, ComputeType.StdConv]:
                self._rearrange_dict[layer.op.weight.name] = (wmem_info.compute_type, wmem_info.filter_group_size)
        for tensor in tensors:
            name = tensor.name
            if tensor.type == TensorType.Constant:
                self._constants[name] = tensor # Do not aware of shared constants
            elif tensor.shared:
                self._shared_features[name] = tensor

        logger.debug("Constants:")
        self._print_status(list(self._constants.keys()))
        logger.debug(f"Shared features (Including I/O):")
        self._print_status(list(self._shared_features.keys()))

    def _run(self) -> None:
        if self._shared_features:
            self._construct_shared_feature_mem()
        if self._constants:
            self._construct_constant_mem()

    def _construct_shared_feature_mem(self):
        import numpy as np
        data = np.zeros(0)
        for name, tensor in self._shared_features.items():
            addr = data.size
            if tensor.type == TensorType.In:
                init_data = tensor.data.reshape(-1)
            else:
                init_data = np.zeros(tensor.data.size, tensor.data.dtype)
            data = np.concatenate([data, init_data]).astype(tensor.data.dtype)
            data = Alignment._align(data, Alignment._addr_align, 0)
            self._memory_info.shared_address_dict[name] = tuple(
                [MemoryType.Shared.value, addr]
            )
        self._memory_info.data_shared_feature = data

    def _construct_constant_mem(
        self,
    ):  # TODO: Weight must be considered with group size... group size must be determined
        import numpy as np

        data = np.zeros(0)
        for name, tensor in self._constants.items():
            addr = data.size
            init_data = tensor.data
            if name in self._rearrange_dict and self._rearrange_dict[name][1] > 1:
                unit_size = (
                    cfg.MIDAP.WMEM.NUM if self._rearrange_dict[name][0] == ComputeType.StdConv
                    else 1
                )
                group_size = self._rearrange_dict[name][1]
                wt = init_data  # N (sequential), W, HC --> G, N/G, W, HC
                new_wt = np.zeros(wt.shape)
                pivot = 0
                logger.debug(
                    f"Tensor {name} is saved while considering filter group size = {group_size}"
                )
                while pivot < wt.shape[0]:
                    gs = min((wt.shape[0] - pivot) // unit_size, group_size)
                    for s in range(unit_size):
                        new_wt[pivot + gs * s : pivot + gs * (s + 1), :, :] = wt[
                            pivot + s : pivot + gs * unit_size : unit_size, :, :
                        ]
                    pivot += gs * unit_size
                init_data = new_wt
            init_data = init_data.reshape(-1)
            data = np.concatenate([data, init_data]).astype(tensor.data.dtype)
            data = Alignment._align(data, Alignment._addr_align, 0)
            self._memory_info.shared_address_dict[name] = tuple(
                [MemoryType.Constant.value, addr]
            )
        self._memory_info.data_shared_constants = data


class MultiIOFrame(Prototype):
    def _preprocess(self) -> None:
        self._constants: OrderedDict[str, Tensor] = OrderedDict()
        self._shared_features: OrderedDict[str, Tensor] = OrderedDict()
        self._input_features: OrderedDict[str, Tensor] = OrderedDict()
        self._output_features: OrderedDict[str, Tensor] = OrderedDict()
        self._rearrange_dict: OrderedDict[str, int] = OrderedDict()
        tensors = self._model.tensors
        for name, wmem_info in self._shared_info.wmem_info_dict.items():
            layer = self._shared_info.model._str2layer[name]
            if wmem_info.compute_type in [ComputeType.DWConv, ComputeType.StdConv]:
                self._rearrange_dict[layer.op.weight.name] = (wmem_info.compute_type, wmem_info.filter_group_size)
        for tensor in tensors:
            name = tensor.name
            if tensor.type == TensorType.Constant:
                self._constants[name] = tensor # Do not aware of shared constants
            elif tensor.shared:
                if tensor.type == TensorType.In:
                    self._input_features[name] = tensor
                elif tensor.type == TensorType.Out:
                    self._output_features[name] = tensor
                else:
                    self._shared_features[name] = tensor

        logger.debug("Constants:")
        self._print_status(list(self._constants.keys()))
        logger.debug(f"Shared features (Including I/O):")
        self._print_status(list(self._shared_features.keys()))

    def _run(self) -> None:
        super(MultiIOFrame, self)._run()
        self._construct_io_feature_mem()

    def _construct_io_feature_mem(self):
        import numpy as np
        data = np.zeros((cfg.MODEL.NUM_FRAMES, 0))
        for name, tensor in self._input_features.items():
            addr = data[0].size
            init_data = tensor.data.reshape(tensor.data.shape[0], -1)
            data = np.concatenate([data, init_data], axis=1).astype(tensor.data.dtype)
            data = Alignment._align(data, Alignment._addr_align, 1)
            self._memory_info.shared_address_dict[name] = tuple(
                [MemoryType.Input.value, addr]
            )
        self._memory_info.data_input_feature = data
        data = np.zeros((cfg.MODEL.NUM_FRAMES, 0))
        for name, tensor in self._output_features.items():
            addr = data.size
            init_data = np.zeros((cfg.MODEL.NUM_FRAMES, tensor.data.size), tensor.data.dtype)
            data = np.concatenate([data, init_data], axis=1).astype(tensor.data.dtype)
            data = Alignment._align(data, Alignment._addr_align, 1)
            self._memory_info.shared_address_dict[name] = tuple(
                [MemoryType.Output.value, addr]
            )
        self._memory_info.data_output_feature = data
