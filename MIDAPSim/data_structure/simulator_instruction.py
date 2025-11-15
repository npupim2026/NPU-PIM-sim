import logging
from midap_backend.wrapper.layer_wrapper import LayerWrapper
from software.compiler.wmem_info import ComputeType
import numpy as np
import copy
import yaml
import math

from config import cfg
from software.generic_op import ConcatOp, ConvOp, Crop, ArithmeticOp
from midap_backend.wrapper.op_wrapper import DWConvWrapper, PoolWrapper, SumWrapper, convert_op_to_wrapper, ConvWrapper, DWWrapper, ArithmeticWrapper, UpBilinearWrapper, MatMulWrapper

from .instruction_components import SLayerInfo, ModuleElement
from .virtual_tensor import VInputTensor, VOutputTensor
from .data import SFMEMInfo, SWMEMInfo, SDataInfo
from collections import OrderedDict

class SimulatorInstruction(object):
    def __init__(self, compiler_input=None):
        self.processing_order = []
        self.addr_dict = {}
        self.data_info_dict = [OrderedDict() for i in range(cfg.MODEL.NUM_FRAMES)]
         # temporal data (output tensors for each layer), SDataInfo
        # self.dram_data = np.zeros(0)
        self.dram_data = None
        self.config = None
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger("debug")
        if compiler_input is not None:
            self.from_compiler_input(**compiler_input)

    def from_compiler_input(self, **kwargs):
        pass

class SimulatorInstructionV2(
    SimulatorInstruction
):  # from compiler_wrapper, 2021-04-14
    def __init__(self, backend_compiler = None, **kwargs):
        super().__init__(**kwargs)
        if backend_compiler is not None:
            self.setup(backend_compiler)
        self.prepare_dict = {}

    def setup(self, backend_compiler):
        self.addr_dict = backend_compiler.addr_dict
        self.dram_data = backend_compiler.dram_data
        ci = backend_compiler.compile_info
        self.compile_info = ci
        self.config = ci.config
        layers = ci.layers
        for key in layers:
            layer = layers[key]
            self.logger.debug(layer.name + ")\n{}".format(layer.processing_info))
            layer_info = self.make_layer_info(layer)
            self.processing_order.append(layer_info)

    def make_layer_info(self, layer : LayerWrapper):
        layer_info = SLayerInfo()
        layer_info.name = layer.name
        tensor_dict = self.compile_info.tensor_dict
        # 1. Input Tensor
        layer_info.input = []
        for winput_tensor in layer.input_tensors:
            input_tensor = VInputTensor(flip_x = winput_tensor.flip_x)
            input_tensor.set_tensor(
                    name = winput_tensor.name,
                    shape = winput_tensor.shape,
                    orig_shape = winput_tensor.orig_shape,
                    init_shape = winput_tensor.init_shape,
                    mapping_type = winput_tensor.mapping_type,
                    offset = winput_tensor.offset,
                    scale = winput_tensor.scale)
            layer_info.input.append(input_tensor)
        # 2. Module
        pi = layer.processing_info
        mm = self.make_module(layer.main_op, layer.main_output, pi.reverse_write)
        layer_info.modules.add(mm)
        if layer.reduction_op is not None:
            rm = self.make_module(layer.reduction_op, layer.reduction_output, pi.reverse_write)
            layer_info.modules.add(rm)
        # 3. Control Info
        ci = layer_info.control_info
        ci.fmem_info = pi.mapping_info
        ci.behavior_info = pi.behavior_info
        # wmem_info 
        ws = pi.wmem_strategy
        wi = ci.wmem_info
        wi.filter_name = ws.filter_name
        wi.load_filter_once = ws.load_filter_once
        wi.filter_group_size = ws.group_size
        wi.prepared = ws.prepared
        wi.reverse_load = ws.reverse_load
        wi.compute_type = ws.compute_type
        wi.reorder_load = ws.reorder_load
        if wi.prepared:
            self.prepare_dict[layer.name].prepare_info = wi
        if ws.prepare_info is not None:
            self.prepare_dict[ws.prepare_info.name] = wi
        # wmem_info: detail
        op = layer.main_op
        if wi.filter_name is not None:
            weight = tensor_dict[0][wi.filter_name].tensor
            wi.filter_size = weight[0,:].size
            wi.num_filters = weight.shape[0]
        if isinstance(op, PoolWrapper) or isinstance(op, SumWrapper) or isinstance(op, UpBilinearWrapper):
            wi.num_filters = input_tensor.shape[-1] // self.config.MIDAP.SYSTEM_WIDTH
            wi.filter_group_size = wi.num_filters
        elif isinstance(op, MatMulWrapper):
            weight = tensor_dict[0][wi.filter_name].tensor
            wi.num_filters = weight.shape[-1] // self.config.MIDAP.SYSTEM_WIDTH
            wi.filter_size = weight.size // wi.num_filters
        elif wi.compute_type == ComputeType.MatMulTrans:
            wi.num_filters = layer_info.input[-1].shape[-3]
        if op.bias is not None:
            wi.bias_name = op.bias
            wi.bias_size = tensor_dict[0][op.bias].tensor.size
        if op.lut is not None:
            wi.lut_name = op.lut
        return layer_info

    def make_module(self, op, output, reverse_write = False):
        tensor_dict = self.compile_info.tensor_dict
        m = ModuleElement()
        m.op = op
        m.name = op.name
        m.quant_info = op.quant_info
        woutput_tensor = output
        output_tensor = VOutputTensor(reverse_write = reverse_write) # virtual, write_on_dram
        output_tensor.set_tensor(
                name = woutput_tensor.name,
                shape = woutput_tensor.shape,
                orig_shape = woutput_tensor.orig_shape,
                init_shape = woutput_tensor.init_shape,
                mapping_type = woutput_tensor.mapping_type,
                offset = woutput_tensor.offset,
                scale = woutput_tensor.scale)
        for i in range(len(self.data_info_dict)):
            if output_tensor.name not in self.data_info_dict[i]:
                output_data = tensor_dict[i][output_tensor.name].tensor
                self.data_info_dict[i][output_tensor.name] = SDataInfo(
                        output_tensor.name,
                        output_data,
                        flip = woutput_tensor.flip_x,
                        )
        m.output.append(output_tensor)
        return m






