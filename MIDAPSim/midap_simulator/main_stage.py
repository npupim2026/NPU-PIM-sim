from data_structure.instruction_components import SModule
from software.network.quant_info import LayerQuantInfo
import numpy as np
import logging

from software.network.types import ActivationType
from midap_backend.wrapper.op_wrapper import ConvPoolWrapper, ConvWrapper, DWWrapper, MaxpoolWrapper, AvgpoolWrapper, ArithmeticWrapper, AddWrapper, MulWrapper, SumWrapper
from .stage import Stage
from .dataflow import generate_dataflow_info
from config import cfg

class FMainStage(Stage): # Cycle - level functional simulation
    def __init__(self, manager):
        super().__init__(manager)
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger("debug")
    
    def initialize(self):
        super().initialize()
        quant = self.manager.config.MODEL.QUANTIZED
        self.quant = quant
        data_type = np.int8 if quant else np.float32
        self.output_buf = np.zeros(self.system_width, dtype = data_type)
        # Add pipeline stage & required components
        self.dataflow_info_buf = [generate_dataflow_info() for _ in range(4)] 
        # Stage 1
        data_type = np.int32 if quant else np.float32
        ### QUANT: int8
        self.wbuf_mem = np.zeros((self.num_wmem, self.system_width), dtype = data_type)
        self.fbuf_mem = np.zeros(self.system_width, dtype = data_type)
        # Stage 2
        self.wbuf = np.zeros((self.num_wmem, self.system_width), dtype = data_type)
        self.fbuf = np.zeros(self.system_width * 2, dtype = data_type)
        self.broadcast_fbuf = np.zeros(self.system_width, dtype = data_type)
        # Stage 3
        ### QUANT: int16
        self.logic_type = 0
        self.use_extended_cim = False
        self.alu_buf = np.zeros([self.num_wmem, self.system_width], dtype = data_type)
        # Stage 4
        ### QUANT int32
        self.adder_count = 0
        self.cim_output_buf = np.zeros(self.num_wmem, dtype = data_type)
        self.ecim_output_buf = np.zeros(self.system_width, dtype = data_type)
        self.csatree_buf = np.zeros(self.num_wmem, dtype = data_type)
        self.accumulator_buf = np.zeros(self.system_width, dtype = data_type)
        # Stage 5
        self.activation = None
        self.add_bias = False
        #self.quant_buf = np.zeros(self.system_width, dtype = np.int32)
        #data_type = np.int16 if quant else np.float32
        self.bias_output_buf = np.zeros(self.num_wmem, dtype = data_type)
        self.act_input_buf = np.zeros(self.num_wmem, dtype = data_type)
        self.act_output_buf = np.zeros(self.num_wmem, dtype = data_type)
        self.e_bias_output_buf = np.zeros(self.system_width, dtype = data_type)
        self.e_act_input_buf = np.zeros(self.system_width, dtype = data_type)
        self.e_act_output_buf = np.zeros(self.system_width, dtype = data_type)
        self.activation_output_buf = np.zeros(self.system_width, dtype = data_type)
        # data_type = np.int32 if quant else np.float32
        self.bias_buf = np.zeros(self.system_width, dtype = data_type)
        self.concurrency = self.num_wmem
        self.dividend = 1
        self.cnt = 0
        self.debug = False

    def setup(self, modules : SModule):
        # if self.debug:
        #     raise RuntimeError
        op = modules[0].op
        self.use_extended_cim = False
        self.concurrency = self.num_wmem
        self.logic_type = 0
        if any([isinstance(op, ArithmeticWrapper), isinstance(op, DWWrapper), isinstance(op, SumWrapper)]):
            self.use_extended_cim = True
            self.concurrency = self.system_width
        if isinstance(op, AvgpoolWrapper):
            self.logic_type = 1
            self.dividend = op.k_h * op.k_w
        elif isinstance(op, SumWrapper):
            self.logic_type = 1 if op.weight is None else 0
            self.dividend = 1
        elif isinstance(op, MaxpoolWrapper):
            self.logic_type = 2
        elif isinstance(op, AddWrapper):
            self.logic_type = 3
        self.adder_count = 0
        self.activation = op.activation
        self.quant_info : LayerQuantInfo = modules[0].quant_info
        assert not self.quant or self.quant_info is not None
        self.add_bias = op.bias is not None
        self.debug = op.name == 'Conv44'

    def run(self, dataflow_info):
        if self.debug:
            self.logger.debug("Input info: {}".format(dataflow_info))
            self.dataflow_info_buf[0] = self.do_load(dataflow_info)
            self.dataflow_info_buf[1] = self.do_broadcast(self.dataflow_info_buf[0])
            self.dataflow_info_buf[2] = self.do_alu(self.dataflow_info_buf[1])
            self.dataflow_info_buf[3] = self.do_adder(self.dataflow_info_buf[2])
            output_dataflow_info = self.do_activator(self.dataflow_info_buf[3])
        else:
            output_dataflow_info = self.do_activator(self.dataflow_info_buf[3])
            self.dataflow_info_buf[3] = self.do_adder(self.dataflow_info_buf[2])
            self.dataflow_info_buf[2] = self.do_alu(self.dataflow_info_buf[1])
            self.dataflow_info_buf[1] = self.do_broadcast(self.dataflow_info_buf[0])
            self.dataflow_info_buf[0] = self.do_load(dataflow_info)
        return output_dataflow_info

    def do_load(self, info):
        if info.phase in [0, 2, 3, 4]:
            return info
        fmem_row = info.fmem_row
        fmem_idx = info.fmem_idx
        wmem_row = info.wmem_row
        if fmem_row > -1:
            self.memory_controller.load_fbuf(self.fbuf_mem, fmem_idx, fmem_row)
        if wmem_row > -1:
            self.memory_controller.load_wbuf(self.wbuf_mem, wmem_row)
        return info

    def do_broadcast(self, info):
        if info.phase in [0, 2, 3, 4]:
            return info
        offset = info.broadcast_offset
        delete_f = info.delete_foffset
        delete_b = info.delete_boffset
        # Shift
        self.fbuf[self.system_width:] = self.fbuf[:self.system_width]
        self.fbuf[0:self.system_width] = self.fbuf_mem[:]
        # Alignment submodule
        if info.fmem_col_broadcast is not None:
            self.broadcast_fbuf[:] = self.fbuf[info.fmem_col_broadcast]
        elif offset > 0:
            self.broadcast_fbuf[:offset] = self.fbuf[-offset:]
            self.broadcast_fbuf[offset:] = self.fbuf[:self.system_width - offset]
        else:
            self.broadcast_fbuf[:] = self.fbuf[:self.system_width]
        if delete_b > 0:
            self.broadcast_fbuf[delete_b:] = np.zeros(
                self.system_width - delete_b)
        if delete_f > 0:
            self.broadcast_fbuf[:delete_f] = np.zeros(delete_f)
        # Load WMEM
        self.wbuf[:, :] = self.wbuf_mem
        self.cnt+=1
        if self.debug:
            self.logger.debug("broadcast_fbuf/{}: {}/{}".format(self.cnt, self.broadcast_fbuf[0:3], np.sum(self.broadcast_fbuf)))
            self.logger.debug("wbuf/{}: {}/{}".format(self.cnt, self.wbuf[0, :3], np.sum(self.wbuf[:1,:], axis=1)))
        return info

    def do_alu(self, info):
        if info.phase in [0, 2, 3, 4]:
            return info
        alu_buf = self.alu_buf[0] if self.use_extended_cim else self.alu_buf
        wbuf = self.wbuf[0] if self.use_extended_cim else self.wbuf
        if self.logic_type == 0:
            alu_buf[:] = np.multiply(self.broadcast_fbuf, wbuf)
        elif self.logic_type in [1, 2]:
            alu_buf[:] = self.broadcast_fbuf[:]
        elif self.logic_type == 3:
            alu_buf[:] = np.add(self.broadcast_fbuf, wbuf)
        #self.logger.debug("alu_buf/{}: {}".format(self.cnt, self.alu_buf[:3, :3]))
        return info

    def do_adder(self, info):
        if info.phase in [0, 2, 3, 4] or info.junk:
            pass
        elif not self.use_extended_cim:
            partial_sum = np.sum(self.alu_buf, axis=1)
            # #self.logger.debug("CSATree - loc: {}, partial_sum: {}".format(info.out_loc, np.sum(partial_sum)))
            if info.reset:
                self.csatree_buf[:] = partial_sum[:]
            else:
                self.csatree_buf = np.add(self.csatree_buf, partial_sum)
            if info.last:
                self.cim_output_buf[:] = self.csatree_buf[:]
        else:
            # Extended CIM Logic
            alu_buf = self.alu_buf[0]
            if info.reset:
                self.accumulator_buf[:] = alu_buf[:]
            elif self.logic_type in [0, 1]: # Avgpool, Depthwise
                self.accumulator_buf[:] = np.add(self.accumulator_buf, alu_buf)
            elif self.logic_type == 2: # Maxpool
                self.accumulator_buf[:] = np.maximum(self.accumulator_buf, alu_buf)
            else:
                raise ValueError("Not a possible scenario")
            if info.last:
                if self.logic_type == 1 and not self.quant:
                    self.ecim_output_buf[:] = np.true_divide(self.accumulator_buf, self.dividend)
                else:
                    self.ecim_output_buf[:] = self.accumulator_buf[:]
            #self.logger.debug("adder_buf/{}: {}".format(self.cnt, self.accumulator_buf[0:3]))
            # #self.logger.debug("Extended CIM - loc: {}, accumulator_buf: {}".format(info.out_loc, self.accumulator_buf[:4]))
        return info

    def do_activator(self, info):
        if info.phase in [0, 2, 3, 4] or info.junk:
            pass
        elif info.last: # 2-way implementation
            self.__add_bias(info.channel_idx)
            self.__do_main_quantize()
            self.__do_truncate(16, self.e_act_input_buf)
            self.__do_truncate(16, self.act_input_buf)
            self.__apply_activation()
            if self.use_extended_cim:
                self.activation_output_buf[:] = self.e_act_output_buf[:]
            else:
                for i in range(0, self.system_width, self.num_wmem):
                    self.activation_output_buf[i:i+self.num_wmem] = self.act_output_buf[:]
            self.__do_second_quantize()
            self.__do_truncate(8, self.activation_output_buf)
            self.output_buf[:] = self.activation_output_buf[:]
            if self.debug:
                offset = info.channel_idx % self.system_width
                self.logger.debug("e_quant_buf/{}: {}".format(self.cnt, self.e_act_input_buf[offset:offset+3]))
                self.logger.debug("quant_buf/{}: {}".format(self.cnt, self.act_input_buf[:3]))
                self.logger.debug("output_buf/{}: {}".format(self.cnt, self.output_buf[offset:offset+3]))
                self.logger.debug("out_loc/{}: {}".format(self.cnt, info.out_loc))
        return info

    def __do_main_quantize(self):
        if self.quant:
            f = self.quant_info.f
            n = self.quant_info.n1
            if f == 1:
                self.act_input_buf[:] = np.right_shift(self.bias_output_buf, n)
                self.e_act_input_buf[:] = np.right_shift(self.e_bias_output_buf, n)
            else:
                self.act_input_buf[:] = np.left_shift(self.bias_output_buf, n)
                self.e_act_input_buf[:] = np.left_shift(self.e_bias_output_buf, n)
        else:
            self.act_input_buf[:] = self.bias_output_buf[:]
            self.e_act_input_buf[:] = self.e_bias_output_buf[:]
    
    def __do_second_quantize(self):
        if self.quant:
            n = self.quant_info.n2
            self.activation_output_buf[:] = np.right_shift(self.activation_output_buf, n)
        else:
            self.activation_output_buf[:] = self.activation_output_buf[:]

    def __add_bias(self, channel_idx):
        if self.add_bias:
            self.memory_controller.load_bbuf(self.bias_buf, channel_idx)
            if self.quant:
                bs = self.quant_info.bs
                self.bias_buf = np.left_shift(self.bias_buf, bs)
                if self.debug:
                    offset = channel_idx % self.system_width
                    self.logger.debug("bbuf/{}: {}".format(self.cnt, self.bias_buf[offset:offset+3]))
            self.e_bias_output_buf[:] = np.add(self.ecim_output_buf, self.bias_buf)
            sel_idx = channel_idx % self.system_width
            self.bias_output_buf[:] = np.add(self.cim_output_buf, self.bias_buf[sel_idx : sel_idx + self.num_wmem])
        else:
            self.e_bias_output_buf[:] = self.ecim_output_buf[:]
            self.bias_output_buf[:] = self.cim_output_buf[:]
    
    def __do_truncate(self, n, buf):
        minimum = int(-(2 ** (n-1)))
        maximum = int((2 ** (n-1)) - 1)
        if self.quant:
            buf[:] = np.where(buf > maximum, maximum, buf)
            buf[:] = np.where(buf < minimum, minimum, buf)
            
    def __apply_activation(self):
        if self.activation == ActivationType.ReLU:
            self.act_output_buf[:] = np.maximum(self.act_input_buf, 0)
            self.e_act_output_buf[:] = np.maximum(self.e_act_input_buf, 0)
        elif self.activation == ActivationType.Linear:
            self.act_output_buf[:] = self.act_input_buf[:]
            self.e_act_output_buf[:] = self.e_act_input_buf[:]
        elif self.activation == ActivationType.Sigmoid and not self.quant:
            self.act_output_buf[:] = 1 / (1 + np.exp(-self.act_input_buf.astype(np.float32)))
            self.e_act_output_buf[:] = 1 / (1 + np.exp(-self.e_act_input_buf.astype(np.float32)))
        elif self.activation == ActivationType.LeakyRelu and not self.quant:
            self.act_output_buf[:] = np.where(self.act_input_buf > 0, self.act_input_buf, self.act_input_buf * 0.1)
            self.e_act_output_buf[:] = np.where(self.e_act_input_buf > 0, self.e_act_input_buf, self.e_act_input_buf * 0.1)
        elif self.quant:
            if self.use_extended_cim:
                raise NotImplementedError(f"ECIM Does not support {self.activation}")
            if self.quant_info.activation_lut is None:
                raise RuntimeError(f"Lookup table must be prepared for {self.activation}")
            x1 = np.right_shift(self.act_input_buf, 8)
            x2 = self.act_input_buf - np.left_shift(x1, 8)
            ab = self.memory_controller.get_lut_items(x1)
            y = ab[:, 0] * x2
            y = np.right_shift(y, 8)
            y = y + ab[:, 1]
            self.act_output_buf[:] = y
        else:
            raise ValueError("Unknown acitvation {}".format(self.activation))

class VMainStage(Stage):
    def __init__(self, manager):
        super().__init__(manager)
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger("debug")

    def initialize(self):
        super().initialize()
        self.skipped_pipeline_stages = 5
        self.buffer = np.zeros([self.num_wmem, self.system_width])
        self.output_buf = np.zeros(self.system_width)
    
    def setup(self, modules):
        op = modules[0].op
        self.use_extended_cim = False
        self.concurrency = self.num_wmem
        self.logic_type = 0
        if any([isinstance(op, ArithmeticWrapper), isinstance(op, DWWrapper), isinstance(op, SumWrapper)]):
            self.use_extended_cim = True
            self.concurrency = self.system_width
        if isinstance(op, AvgpoolWrapper):
            self.logic_type = 1
        elif isinstance(op, SumWrapper):
            self.logic_type = 1 if op.weight is None else 0
            self.dividend = 1
        elif isinstance(op, MaxpoolWrapper):
            self.logic_type = 2 
        elif isinstance(op, AddWrapper):
            self.logic_type = 3
        self.add_bias = op.bias is not None

    def run(self, info):
        if info.phase in [0, 2, 3, 4]:
            return info
        fmem_row = info.fmem_row
        fmem_idx = info.fmem_idx
        wmem_row = info.wmem_row
        if fmem_row > -1:
            self.memory_controller.load_fbuf(self.buffer[0], fmem_idx, fmem_row)
        if wmem_row > -1:
            self.memory_controller.load_wbuf(self.buffer, wmem_row)
        if info.last and self.add_bias:
            self.memory_controller.load_bbuf(self.buffer[0], 0)
        return info


