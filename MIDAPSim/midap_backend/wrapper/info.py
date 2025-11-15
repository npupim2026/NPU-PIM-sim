import math
from typing import Any, List
from collections import defaultdict, deque

import numpy
from midap_backend.wrapper.tensor_wrapper import TensorWrapper
from software.network.types import OpType

from data_structure.attrdict import AttrDict
from data_structure.data import SFMEMInfo
from software.compiler.layer_compile import LayerInfo
from software.compiler.layer_compile.behavior import Action


class WriteInfo(object):
    def __init__(self, write_type, write_unit, write_shape, write_crit, write_offset=0):
        self.write_type = write_type
        # self.write_unit = write_unit
        self.write_shape = write_shape
        self.write_crit = write_crit
        self.data_offset = write_offset  # DRAM Offset

    @property
    def type(self):
        return self.write_type

    @type.setter
    def type(self, value):
        self.write_type = value

    @property
    def shape(self):
        return self.write_shape

    @shape.setter
    def shape(self, value):
        self.write_shape = value

    @property
    def crit(self):
        return self.write_crit

    @crit.setter
    def crit(self, value):
        self.write_crit = value


class QuantInfo(object):
    def __init__(self, shift_flag, main_shift, act_shift, bias_shift, lut = None):
        self.shift_flag: int = shift_flag
        self.main_shift : int = main_shift
        self.act_shift: int = act_shift
        self.bias_shift: int = bias_shift
        self.activation_lut = None
        if lut is not None:
            self.activation_lut = numpy.array(lut, dtype = numpy.int32)

    @property
    def lut(self):
        return self.activation_lut
        
    @property
    def value(self):
        return self.shift_flag, self.main_shift, self.act_shift, self.bias_shift

    @property
    def f(self):
        return self.shift_flag

    @property
    def n1(self):
        return self.main_shift
    
    @property
    def n2(self):
        return self.act_shift

    @property
    def bs(self):
        return self.bias_shift

    @property
    def bshift(self):
        return self.bias_shift


class TensorInfo:
    def __init__(self, name, tensor, fixed=False, init=False):
        self.name = name
        self.tensor = tensor
        self.fixed = fixed
        self.init = init

class ProcessingInfo:
    def __init__(self):
        self.mapping_info = SFMEMInfo()
        self.behavior_info = SBehaviorInfo()
        self.write_logic_type = 0
        # write_logic_type
        # 0 : Do not use
        # 1 : Save Whole ZY Plane & Write
        # 2 : Save Z'Y Plane & Write (Z'=64)
        # 3 : Save Z'YX' Plane & Write (Z'=64, X': depends on behavior_info)
        self.write_on_dram_pivot = 0xFFFF
        self.reverse_write = False
        self.wmem_strategy = AttrDict(
            dict(
                filter_name=None,
                compute_type=None,
                load_filter_once=False,
                group_size=1,
                prepare_info=None,
                prepared=False,
                reverse_load=False,
                reorder_load=False,
            )
        )

    def __repr__(self):
        rep_str = "write_type: {}\n".format(self.write_logic_type)
        rep_str += "write_dram_pivot: {}\n".format(self.write_on_dram_pivot)
        rep_str += "Z_tile (in, out): {}\n".format(self.behavior_info.z_tile)
        rep_str += "reverse_write: {}\n".format(self.reverse_write)
        rep_str += "wmem_Strategy: {}\n".format(self.wmem_strategy)
        return rep_str


class Behavior(list):
    def __init__(self, behavior_type, in1, in2 = None, in3 = None):
        super().__init__([behavior_type, in1, in2, in3])
        # Condition : [-1, k in [0 ~ N-1], N = # of FMEM banks - 1]
        # -1 : After the finish of previous behavior
        # k : right after processing x==k th input mapping (corresponding input)
        # --> should be changed to index-based condition
        # behavior : [load/process, name, k]
        # load: load k th input mapping of corresponding data
        # process: process ~ kth input mapping of current layer
        self.type = behavior_type
        if behavior_type == "LOAD":
            self.cond = in1
            self.input_name = in2
            self.index = in3
        elif behavior_type == "PROCESS":
            self.idx = in1
            self.min_x = in2
            self.max_x = in3
        self.write_info = None


class SBehaviorInfo(list):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_z_tile = 1
        self.out_z_tile = 1
        self.min_y, self.max_y = [0, 0]
        # self.write_info = None
        self.main_op, self.input_tensor, self.input_mapping = [
            None,
            None,
            None,
        ]  # Temporal data

    def from_layer_info(self, layer_info: LayerInfo, input_tensor : TensorWrapper):
        # print(layer_info.layer.name)
        self.input_tensor = input_tensor
        self.main_op = layer_info.op
        # stationary = layer_info.stationary
        #if layer_info.op.type == OpType.HostProcess:
        #    return
        behaviors = layer_info.behavior
        # self.input_mapping = list(filter(lambda dt: dt.data.tensor == layer_info.ordered_input_tensors[0], layer_info.mapping.input))
        loaded_inputs = layer_info.num_init_input()
        counter = defaultdict(int)
        idx_dict = defaultdict(deque)
        for mp in layer_info.mapping.input:
            cnt = counter[mp.data.tensor]
            idx_dict[mp.data].append(cnt)
            counter[mp.data.tensor] = cnt + 1
        processed_inputs = 0
        self.min_y, self.max_y = self.set_min_max_y()
        for b in behaviors:
            action_type = b.action
            if action_type == Action.LOAD:
                for dt in b.data:
                    input_name = dt.tensor.name
                    self.append(Behavior("LOAD", -1, input_name, idx_dict[dt][0]))
                loaded_inputs += b.num_banks
            elif action_type == Action.PROCESS:
                num_banks = b.num_banks
                idx = processed_inputs + num_banks
                min_x, max_x = self.get_min_max_x(
                    b.data[0].pivot, b.data[-1].last_x, idx < loaded_inputs, processed_inputs == 0, idx == len(layer_info.mapping.input)
                )
                if min_x <= max_x:
                    for d in b.data[:-1]:
                        idx_dict[d].popleft()   # Never used again in current layer
                    self.append(Behavior("PROCESS", idx_dict[b.data[-1]].popleft(), min_x, max_x))
                # for input_id in range(processed_inputs, idx):
                #     load_flag = self.input_mapping[input_id][-1]
                #     if load_flag:
                #         cond_x = self.get_output_x(self.input_mapping[input_id + 1][1][0])
                #         self.append(Behavior('LOAD', cond_x, input_name, loaded_inputs))
                #         loaded_inputs += 1
                # TODO: Load Flag
                processed_inputs = idx
            elif action_type == Action.SYNC:
                self.append(Behavior("SYNC", b.id))
            elif action_type == Action.WAIT:
                self.append(Behavior("WAIT", b.id))

    def set_min_max_y(self):
        main_op = self.main_op
        input_tensor = self.input_tensor
        if not input_tensor:
            return 0, 0
        y_min = 0
        y_max = input_tensor.shape[1]
        y_min = -main_op.pad_t
        y_max += main_op.pad_b
        y_max -= (main_op.k_h -1) * main_op.dilation + 1
        return y_min, y_max

    def get_min_max_x(self, head, tail, next_on_chip=False, first = False, last = False):
        main_op = self.main_op
        scale = self.input_tensor.scale[0]
        offset = self.input_tensor.offset[0]
        input_shape = self.input_tensor.shape
        x_min = max(0, (head - offset) * scale)
        x_max = max(0, (tail - offset) * scale - 1)
        # if isinstance(main_op, ConvPoolOpBase):
        x_limit = input_shape[0] + main_op.pad_r - 1 - (main_op.k_w-1)  * main_op.dilation
        if first:
            x_min -= main_op.pad_l
        if last:
            x_max += main_op.pad_r
        if not next_on_chip:
            x_max -= (main_op.k_w - 1) * main_op.dilation
        remain = (x_min + main_op.pad_l) % main_op.stride
        if remain > 0:
            x_min = x_min + remain
        x_max = min(x_max, x_limit)
        return x_min, x_max

    def get_output_x(self, x):
        main_op = self.main_op
        return math.ceil((x + main_op.pad_l) / main_op.stride)

    @property
    def z_tile(self):
        return self.in_z_tile, self.out_z_tile

    def __repr__(self):
        orig_str = super().__repr__()
        return "List of [condition, behavior_type, input_name, input_idx] = " + orig_str
