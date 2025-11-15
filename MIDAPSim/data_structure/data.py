from collections import OrderedDict
from software.network.virtual_tensor import VirtualTensor
from software.compiler.layer_compile.mapping import Mapping
from typing import List

import numpy as np
from software.compiler.layer_compile import LayerInfo
from software.generic_op import ArithmeticOp, ConvOp, PoolOp


class SDataInfo(object):
    def __init__(self, name, data, flip=False, require_dram_space=True):
        self.shape = data.shape
        self.name = name
        self.data = data
        self.flip = flip
        # if flip:
        #    self.data = np.flip(self.data, axis=0)
        self.compare_data_logical = np.zeros(self.shape)
        self.compare_data_memory = np.zeros(self.shape)
        self.compare_data = {'logical': self.compare_data_logical, 'memory': self.compare_data_memory}
        self.require_dram_space = (
            require_dram_space  # Not in use.. for further implementation
        )
        self.diff_arr = {'logical': np.zeros(self.shape), 'memory': np.zeros(self.shape)}
        self.diff_cnt = {'logical': 0, 'memory': 0}

    def get_compare_data(self):
        return self.compare_data

    def check_result(self, offset=None, shape=None, name=None):
        if offset == None:
            offset = (0, 0, 0)
        if shape == None:
            shape = self.shape
        if name == None:
            name = self.name
        end_offset = [o + s for o, s in zip(offset, shape)]
        sx, sy, sz = offset
        dx, dy, dz = end_offset
        n = self.data[sx:dx, sy:dy, sz:dz]
        pl = self.compare_data_logical[sx:dx, sy:dy, sz:dz]
        pm = self.compare_data_memory[sx:dx, sy:dy, sz:dz]
        diff_logic = np.abs(n - pl)
        diff_mem = np.abs(n - pm)
        abs_arr_logic = np.abs(n) + np.abs(pl)
        abs_arr_logic = np.where(abs_arr_logic > 0, abs_arr_logic, 1)
        abs_arr_mem = np.abs(n) + np.abs(pm)
        abs_arr_mem = np.where(abs_arr_mem > 0, abs_arr_mem, 1)
        diff_ratio_logic = np.true_divide(diff_logic, abs_arr_logic)
        diff_ratio_mem = np.true_divide(diff_mem, abs_arr_mem)
        diff_arr_logic = np.where(diff_ratio_logic < 0.01, 0, 1)
        diff_arr_mem = np.where(diff_ratio_mem < 0.01, 0, 1)
        diff_value_logic = np.sum(diff_arr_logic)
        diff_value_mem = np.sum(diff_arr_mem)
        ret_str = "Function Simulation result - layer: {}, diff(logical): {} / {}, diff(memory): {} / {}".format(
            name, diff_value_logic, diff_arr_logic.size, diff_value_mem, diff_arr_mem.size
        )
        self.diff_arr = {'logical': diff_arr_logic, 'memory': diff_arr_mem}
        self.diff_cnt = {'logical': diff_value_logic, 'memory': diff_value_mem}
        diff_ratio = {'logical': diff_value_logic / self.data.size, 'memory': diff_value_mem / self.data.size}
        return diff_ratio, ret_str


class DataMapping(list):
    def __init__(self, mem_id, head, tail):
        super().__init__([mem_id, head, tail])
        self.mem_id = mem_id
        self.head = head
        self.tail = tail
        # or head idx & size?


class MappingInfo(list):
    def __init__(self, name, shape, init_shape=None):
        self.name = name
        self.shape = shape
        self.init_shape = shape if init_shape is None else init_shape
        self.write_on_dram_pivot = 0
        self.yz_plane_size = shape[-1] * shape[-2]

    def add(self, idx, head, tail):
        mapping = DataMapping(idx, head, tail)
        self.append(mapping)

    def __repr__(self):
        return f"Shape: {self.shape}, Write Pivot : {self.write_on_dram_pivot}\nMapping: {super().__repr__()}"


class SFMEMInfo(object):
    def __init__(self, **kwargs):
        self.input_mapping = OrderedDict()  # MappingInfo
        self.output_mapping = OrderedDict()  # Where to write?

    def add_layer_info(self, layer_info: LayerInfo):
        mapping_info = layer_info.mapping
        for info in mapping_info.input:
            # fmem_idx, (head, tail), flag = info
            input_name = info.data.tensor.name
            input_shape = (1, *info.data.tensor.shape[1:])
            input_init_shape = (1, *info.data.tensor.init_shape[1:])
            fmem_idx = info.bank
            head = info.data.pivot
            tail = info.data.last_x
            if input_name not in self.input_mapping:
                self.input_mapping[input_name] = MappingInfo(input_name, input_shape, input_init_shape)
            self.input_mapping[input_name].add(fmem_idx, head, tail)
        self.add_output_mapping(layer_info.layer.out_vtensor, mapping_info.output)
        # output_name = layer_info.layer.out_vtensor.name
        # output_shape = layer_info.layer.out_vtensor.orig_shape
        # if output_name not in self.output_mapping:
        #     self.output_mapping[output_name] = MappingInfo(output_name, output_shape)
        # for info in mapping_info.output:
        #     output_name = info.data.tensor.name
        #     input_shape = info.data.tensor.shape
        #     fmem_idx = info.bank
        #     head = info.data.pivot
        #     tail = info.data.last_x
        #     if output_name not in self.output_mapping:
        #         self.output_mapping[output_name] = MappingInfo(
        #             output_name, output_shape
        #         )
        #     self.output_mapping[output_name].add(fmem_idx, head, tail)
        #     # For compile-time
        #     pivot = self.output_mapping[output_name].write_on_dram_pivot
        #     self.output_mapping[output_name].write_on_dram_pivot = max(pivot, tail)

    def add_output_mapping(self, vtensor : VirtualTensor, mapping: List[Mapping]):
        output_name = vtensor.name
        output_shape = (1, *vtensor.orig_shape[1:])
        output_init_shape = (1, *vtensor.orig_init_shape[1:])
        if output_name not in self.output_mapping:
            self.output_mapping[output_name] = MappingInfo(output_name, output_shape, output_init_shape)
        for info in mapping:
            output_name = info.data.tensor.name
            input_shape = (1, *info.data.tensor.shape[1:])
            fmem_idx = info.bank
            head = info.data.pivot
            tail = info.data.last_x
            if output_name not in self.output_mapping:
                self.output_mapping[output_name] = MappingInfo(
                    output_name, output_shape
                )
            self.output_mapping[output_name].add(fmem_idx, head, tail)
            # For compile-time
            pivot = self.output_mapping[output_name].write_on_dram_pivot
            self.output_mapping[output_name].write_on_dram_pivot = max(pivot, tail)


class SWMEMInfo(object):
    def __init__(self, **kwargs):
        self.filter_name = None  # Instead of address
        self.bias_name = None  # Instead of address
        self.bias_size = 0
        self.lut_name = None
        self.filter_size = 0  # Filter size
        self.num_filters = 0
        self.compute_type = 0
        self.load_filter_once = False
        self.filter_group_size = 1
        self.prepare_info = None  # data name, type, size
        self.prepared = False
        self.reverse_load = False
