import math
from typing import List
import numpy as np
import logging
from midap_backend.wrapper.layer_wrapper import LayerWrapper
from midap_backend.wrapper.op_wrapper import OpWrapper

from software.generic_op import ConvOp, UpsampleOp, Crop, ArithmeticOp, PoolOp, ConvPoolOpBase
from config import cfg
from midap_backend.wrapper.info import QuantInfo, SBehaviorInfo

from .virtual_tensor import VInputTensor, VOutputTensor
from .data import SFMEMInfo, SWMEMInfo

class SLayerInfo(object):
    def __init__(self, **kwargs):
        self.input : List[VInputTensor] = []
        self.name = None
        self.modules = SModule()
        self.control_info = SControlInfo()

    def from_midap_layer(self, midap_layer):
        # Set Input
        self.name = midap_layer.name
        self.set_input(midap_layer)
        self.modules.from_midap_layer(midap_layer)
        self.control_info.from_midap_layer(midap_layer, self.input)
    
    def set_input(self, midap_layer):
        #input tensor
        scale = [1, 1, 1]
        offset = [0, 0, 0]
        mapping_type = 'default'
        if isinstance(midap_layer.mapping_op, UpsampleOp):
            mapping_type = 'valid' if midap_layer.mapping_op.algorithm.lower() == 'zero' else 'linear'
        elif isinstance(midap_layer.mapping_op, Crop) or midap_layer.mapping_func is not None:
            mapping_type = 'linear'
        if mapping_type != 'default':
            scale = [midap_layer.scale_w, midap_layer.scale_h, 1]
            offset = [midap_layer.x_offset[0], midap_layer.y_offset[0], 0]
        data = midap_layer.input[0]
        data_name = data.output_name
        input_orig_shape = data.output_tensor.shape
        w, h, c = input_orig_shape
        w -= sum(midap_layer.x_offset)
        h -= sum(midap_layer.y_offset)
        input_shape = [w, h, c]
        input_shape = [i*j for i, j in zip(scale, input_shape)]
        input_tensor = VInputTensor(flip_x = midap_layer.control_info.input_flip)
        input_tensor.set_tensor(
                name = data_name,
                shape = input_shape,
                orig_shape = input_orig_shape,
                init_shape = input_shape,
                mapping_type = mapping_type,
                offset = offset,
                scale = scale)
        #Input tensor for second input cannot be virtualized yet...
        self.input = [input_tensor]

    def __repr__(self):
        s = "<<<Layer Input Information>>>\n"
        for tensor in self.input:
            s += str(tensor) + '\n'
        s += "<<<Processing Information>>>\n"
        s += "Process: {}\n".format(self.name)
        for idx, module in enumerate(self.modules):
            s += 'Op {}: '.format(idx + 1) + str(module.op)
            if len(module.output) > 0:
                s += 'Output: {}\n'.format(module.output[0])
        s += "Action: {} \n".format(self.control_info.behavior_info)
        s += "Expected input mapping: {}\nExpected output mapping: {}".format(self.control_info.get_input_mapping(), self.control_info.get_output_mapping())
        return s


class SModule(object):
    def __init__(self):
        super().__init__()
        self.__modules : List[ModuleElement] = []

    def add(self, element):
        if not isinstance(element, ModuleElement):
            raise ValueError("Undefined element type")
        self.__modules.append(element)

    def from_midap_layer(self, midap_layer):
        # Main module
        main_module = ModuleElement(midap_layer)
        self.add(main_module)
        # Reduction module
        if midap_layer.have_reduction_layer:
            reduction_module = ModuleElement(midap_layer.next[0])
            self.add(reduction_module)

    def get_output(self, module_idx = -1, output_idx = 0):
        self[module_idx].get_output(output_idx)

    def __getitem__(self, idx):
        return self.__modules[idx]

    def __len__(self):
        return len(self.__modules)

class ModuleElement(object):
    def __init__(self):
        self.op : OpWrapper = None
        self.processing_type = None
        self.output : List[VOutputTensor] = []
        self.name = None
        self.quant_info = None
    
    def get_output(self, output_idx = 0):
        return self.output[output_idx]

    def from_midap_layer(self, midap_layer):
        self.op = midap_layer.main_op
        self.name = self.op.name
        if any([isinstance(self.op, ArithmeticOp),
                isinstance(self.op, PoolOp),
                self.op.type in ['Depthwise']]):
            self.processing_type = 'extended'
        elif isinstance(self.op, ConvOp):
            self.processing_type = 'default'

        control_info = midap_layer.control_info
        virtual = False
        if midap_layer.have_reduction_layer and not midap_layer.write_on_dram:
            virtual = True
        data_name = midap_layer.output_name
        # for concat
        offset = [0, 0, 0]
        orig_shape = midap_layer.output_tensor.shape
        offset_temp = list(zip(offset, orig_shape))
        offset_temp[midap_layer.offset_axis] = midap_layer.offset
        shape = [x[1] - x[0] for x in offset_temp]
        offset = [x[0] for x in offset_temp]
        # for upsampling, zero insertion
        sub_op = midap_layer.sub_op
        scale = [1, 1, 1]
        # In future: sub_op should integrate whole output tensor virtualization formats
        # @@@ I think that concat should be included in sub_op
        mapping_type = 'default' if sum(offset) == 0 else 'linear'
        if isinstance(sub_op, UpsampleOp):
            if sub_op.algorithm == 'Zero':
                mapping_type = 'zero'
            else:
                mapping_type = 'linear'
            scale = [sub_op.k_w, sub_op.k_h, 1]
            # shape = [size//scale for size, scale in zip(shape, scale)]
        elif isinstance(sub_op, Crop):
            raise ValueError("Crop for sub_op is a weird case.. should not occur")

        tensor = VOutputTensor(reverse_write = control_info.reverse_write, write_on_dram = midap_layer.write_on_dram, virtual = virtual)
        tensor.set_tensor(
                name = data_name,
                shape = shape,
                orig_shape = orig_shape,
                init_shape = shape,
                mapping_type = mapping_type,
                offset = offset,
                scale = scale)
        self.output.append(tensor)
        # Extra outputs: must be saved with un-flipped form
        reverse_write = not control_info.input_flip
        for info in midap_layer.extra_output_info:
            extra_tensor = VOutputTensor(reverse_write = reverse_write)
            extra_data_name = info.name
            extra_orig_shape = info.shape
            extra_tensor.set_tensor(extra_data_name, shape, extra_orig_shape, shape, 'linear', offset, scale)
            self.output.append(extra_tensor)

    def set_quant_info(self, scale, bias_scale = 0):
        # print("Layer: {}, scale : {}, {}".format(self.name, scale, bias_scale))
        if scale < 0:
            sign = 1
            n = int(-scale)
        else:
            sign = 0
            n = int(scale)
        bs = int(bias_scale)
        self.quant_info = QuantInfo(sign, n, bs)


class SControlInfo(object):
    def __init__(self):
        self.fmem_info = SFMEMInfo()
        self.wmem_info = SWMEMInfo()
        self.behavior_info = SBehaviorInfo()

    def get_input_mapping(self, name = None):
        if not self.fmem_info.input_mapping:
            return []
        if name is None:
            name = list(self.fmem_info.input_mapping.keys())[0]
        return self.fmem_info.input_mapping[name]

    def get_output_mapping(self, name = None):
        if not self.fmem_info.output_mapping:
            return []
        if name is None:
            name = list(self.fmem_info.output_mapping.keys())[0]
        return self.fmem_info.output_mapping[name]
