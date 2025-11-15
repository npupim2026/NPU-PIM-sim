from __future__ import annotations

import logging
from functools import reduce
from typing import TYPE_CHECKING, Dict, List

import attr
from config import cfg
from logger import init_logger
from software.compiler.layer_compile.layer_info import LayerInfo
from software.network.tensor import Tensor
from software.network.types import OpType, TensorType
from software.system_compiler.core_info import CoreInfo

from .strategy import Strategy

if TYPE_CHECKING:
    from software.system_compiler.system_info import SystemInfo


__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("Local Memory Compile", logging.INFO)


class LocalMemoryCompile(Strategy):
    def _print_status(self, curr):
        def _print_func(l):
            return [repr(x) for x in l]

        logger.debug("====================================================")
        l = _print_func(curr)
        logger.debug(f"{curr}")
        logger.debug("====================================================")

    def _initialize(self, info: SystemInfo, core_info: CoreInfo) -> None:
        self._mem_info = info.memory_info
        self._core_info = core_info

    def _preprocess(self) -> None:
        # Define shared items
        pass

    def compile(self, info: SystemInfo) -> SystemInfo:  # Per core processing
        new_core_info = []
        for core_info in info.core_info:
            self._initialize(info, core_info)
            self._preprocess()
            self._run()
            self._postprocess()
            new_core_info.append(self._core_info)
            info.memory_info.local_memory_info_dict[
                core_info.core_id
            ] = self._core_info.local_memory_info
        info.core_info = new_core_info
        return info

    def _run(self) -> None:
        pass

    def _postprocess(self) -> None:
        pass


@attr.s(init=True, slots=True, repr=False)
class BufInfo(object):
    tensor: Tensor = attr.ib()
    head: int = attr.ib()
    size: int = attr.ib()

    @property
    def tail(self):
        return self.head + self.size

    def __repr__(self):
        return f"({self.tensor.name}: {self.head}, {self.size})"


class Prototype(LocalMemoryCompile):
    def _preprocess(self):  # Update WMEM Information
        self._layers = self._core_info.compile_info.layers
        self._remain_pivot_dict: Dict[Tensor, LayerInfo] = dict()
        self._write_pivot_x_dict: Dict[Tensor, int] = dict()
        last_wmem_use_layer = None
        for layer_info in self._layers:
            if layer_info.dummy:
                continue
            self.__update_pivot(layer_info)
            last_wmem_use_layer = self.__update_wmem_info(
                layer_info, last_wmem_use_layer
            )

    def __update_pivot(self, layer_info: LayerInfo):
        mapping = layer_info.mapping
        input_tensors = layer_info.ordered_input_tensors
        im = {tensor: list(filter(lambda m: m.data.tensor == tensor, mapping.input)) for tensor in input_tensors}
        ni = {tensor: layer_info.num_init_input(tensor) for tensor in input_tensors}
        for tensor in input_tensors:
            exist = False
            for dt in im[tensor][:ni[tensor]]:
                if tensor == dt.data.tensor:
                    exist = True
                    break
            if not exist:
                self._write_pivot_x_dict[tensor] = 0
                self._remain_pivot_dict[tensor] = layer_info
            else:
                for dt in im[tensor][ni[tensor]:]:
                    if tensor == dt.data.tensor:
                        x = dt.data.pivot
                        self._remain_pivot_dict[tensor] = layer_info
                        if tensor in self._write_pivot_x_dict:
                            self._write_pivot_x_dict[tensor] = min(
                                self._write_pivot_x_dict[tensor], x
                            )
                        else:
                            self._write_pivot_x_dict[tensor] = x
                        break
        if layer_info.out_tensor.is_shared_tensor:
            self._write_pivot_x_dict[layer_info.out_tensor] = 0

    # It might be
    def __update_wmem_info(
        self, layer_info: LayerInfo, last_wmem_use_layer: (str | None)
    ) -> (str | None):
        layer = layer_info.layer
        filter_name = None
        if layer_info.op.weight is not None:
            filter_name = layer_info.op.weight.name
        elif len(layer_info.ordered_input_tensors) > 1 and layer.op.type != OpType.WeightedSum:
            filter_name = layer_info.ordered_input_tensors[-1].name
        else:
            return last_wmem_use_layer
        wmem_info_dict = self._core_info.compile_info.wmem_info_dict
        wmem_info_dict[layer.name].filter_name = filter_name
        if last_wmem_use_layer is not None:
            if self._mem_info.spm_info.in_use:
                pass
            elif layer_info.op.weight is not None or filter_name != last_wmem_use_layer:
                wmem_info_dict[last_wmem_use_layer].prepare_info = layer.name
                wmem_info_dict[layer.name].prepared = True
        return layer.name

    def _run(self):
        lmi = self._core_info.local_memory_info
        buf_info: List[BufInfo] = []
        max_buf_size = 0
        addr_dict = lmi.address_dict
        pivot_x_dict = lmi.write_mem_pivot_dict
        align_unit = cfg.DRAM.ADDRESS_ALIGN
        for layer_info in self._layers:  # TODO : Capsulization...
            if layer_info.dummy:
                continue
            tensor = layer_info.out_tensor
            if tensor in self._remain_pivot_dict:
                pivot_x_dict[layer_info.layer.name] = self._write_pivot_x_dict[tensor]
            elif tensor.is_shared_tensor:
                pivot_x_dict[layer_info.layer.name] = 0
            else:
                pivot_x_dict[layer_info.layer.name] = tensor.width
            if all(
                [
                    reduce(
                        lambda acc, cur: acc and cur.tensor != tensor, buf_info, True
                    ),
                    (tensor in self._remain_pivot_dict),
                    tensor.type == TensorType.Temporal,
                ]
            ):
                pivot_x = self._write_pivot_x_dict[tensor]
                t_size = (
                    tensor.width - pivot_x
                ) * tensor.yz_plane_size  # Write only partial data on the buf
                t_size = (t_size + align_unit - 1) // align_unit * align_unit
                idx = 0
                addr = 0
                for info in buf_info:
                    if t_size <= info.head - addr:
                        break
                    else:
                        addr = info.tail
                        idx += 1
                new_info = BufInfo(tensor, addr, t_size)
                buf_info = buf_info[:idx] + [new_info] + buf_info[idx:]
                addr_dict[tensor.name] = addr - (
                    pivot_x * tensor.yz_plane_size
                )  # Register relative address considering write pivot..
                max_buf_size = max(max_buf_size, buf_info[-1].tail)
            if buf_info:
                buf_info = list(
                    filter(
                        lambda x: self._remain_pivot_dict[x.tensor] != layer_info,
                        buf_info,
                    )
                )
                logger.debug(f"Layer {layer_info.layer} Updated Buf Info")
                self._print_status(buf_info)
        lmi.required_size = max_buf_size
