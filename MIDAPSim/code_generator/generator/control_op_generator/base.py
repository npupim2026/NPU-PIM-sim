from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import logging.config

import numpy as np
import math

from config import cfg

from midap_simulator.control_logic.base import ControlLogicBase
from midap_backend.wrapper.op_wrapper import ConvPoolWrapper, ConvWrapper, DWWrapper, PoolWrapper, DWConvWrapper, ArithmeticWrapper, SumWrapper, UpBilinearWrapper
from midap_backend.wrapper.info import WriteInfo

from data_structure.instruction_components import SLayerInfo

from code_generator.env import rule
from code_generator.env.rule import TriggerType
from code_generator.env.sfr import reg
from code_generator.env.opcode import set_reg, reset_reg, wait_reg_eq, wait_reg_zero, wait_fin

class ControlOpGenerator(ControlLogicBase):
    prohibited_reg_ids = list(reg.DMA.values())
    def __init__(self, generator):
        # Initialize System Configuration
        config = generator.config
        self.config = config
        self.system_width = config.MIDAP.SYSTEM_WIDTH
        self.num_wmem = config.MIDAP.WMEM.NUM
        self.num_fmem = config.MIDAP.FMEM.NUM
        self.concurrency = self.num_wmem
        self.opcode_generator = generator
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('gen')
        self.last_trigger_id = 0

    @property
    def registers(self):
        return self.opcode_generator.registers

    @property
    def memory_op_generator(self):
        return self.opcode_generator.memory_op_generator

    @property
    def svar(self):
        return self.opcode_generator.svar
    
    def gen_init_code(self):
        init_code = []
        init_code += reset_reg(reg.trigger_csg)
        return init_code

    def set_reg(self, reg_id, value, force = False):
        if not isinstance(reg_id, int):
            return self.set_reg_bits(reg_id.no, value, reg_id.bits, force)
        if value < 0:
            raise ValueError("Setting value must be >= 0")
        if reg_id in self.prohibited_reg_ids:
            raise ValueError("prohibited_reg_id: {}".format(reg_id))
        if self.registers.get_value(reg_id) == value and not force:
            return []
        else:
            self.registers.set(reg_id, value)
            return set_reg(reg_id, value)
    
    def set_reg_bits(self, reg_id, values_arr, bits_arr, *args, **kwargs):
        if sum(bits_arr) > self.svar.register_bits:
            raise ValueError("sum of bits must be smaller than register bits, {}".format(self.svar.register_bits))
        value = 0
        for val, bits in reversed(list(zip(values_arr, bits_arr))):
            value = value << bits
            value += val
        return self.set_reg(reg_id, value, *args, **kwargs)

    def trigger_and_wait(self, trigger_value):
        # if trigger_value == 0:
        #     raise ValueError("Invalid Triggering")
        trigger_id = 4 + ((self.last_trigger_id + 1) % 4)
        self.last_trigger_id = trigger_id
        trigger = self.set_reg(reg.trigger_csg, [trigger_value, trigger_id], force = True)
        wait_reg_id = reg.feedback_from_csg
        wait = wait_reg_eq(wait_reg_id, trigger_id)
        return trigger + wait

    def setup(self, layer_info : SLayerInfo):
        super().setup(layer_info)
        self.layer_info = layer_info
        #init_code = self.gen_init_code()
        code = self._setup_processing_info(layer_info)
        code += self.gen_reset_code() 
        return code

    def _setup_processing_info(self, layer_info : SLayerInfo):
        # Setup code generator for convpool operator
        main_op = self.main_op
        self.gen_dedicated_code = self.default_worker
        lp_setup = self._setup_lp(layer_info)
        param_setup = self._register_fixed_params(layer_info)
        if isinstance(main_op, ConvWrapper) and self.input_tensor.shape[-1] % self.system_width > 0:
            gen = self.gen_conv_yz_code
            self.logger.info("Code Generator: Conv-YZ")
        elif isinstance(main_op, ConvPoolWrapper):
            if not self.input_tensor.is_vt:
                gen = self.gen_conv_z_code
                self.logger.info("Code Generator: Conv-Z")
            else:
                gen = self.gen_conv_z_tv_code
                self.logger.info("Code Generator: Conv-Z-TV")
        elif isinstance(main_op, ArithmeticWrapper):
            if not self.input_tensor.is_vt:
                gen = self.gen_arithmetic_code
            else:
                gen = self.gen_arithmetic_tv_code
        elif isinstance(main_op, SumWrapper):
            if not self.input_tensor.is_vt:
                gen = self.gen_weighted_sum_code
            else:
                gen = self.gen_weighted_sum_tv_code
        else:
            raise RuntimeError("Unknown op: {}".format(main_op))
        self.gen_dedicated_code = gen
        return lp_setup + param_setup

    def _setup_lp(self, layer_info: SLayerInfo, reduction = False):
        op = self.main_op if not reduction else self.reduction_op
        idx = 0 if not reduction else 1
        cim_type = rule.cim_type(op) if not reduction else 0
        act_type = rule.act_type(op.activation)
        bias_type = rule.bias_type(op)
        if layer_info.modules[idx].quant_info is None:
            raise RuntimeError("All layers must have quant info")
        qi = layer_info.modules[idx].quant_info
        quant_type, main_shift, act_shift, bias_shift = qi.value
        reduction_type = 0
        if len(layer_info.modules) > 1:
            reduction_type = 1
        set_lps = self.set_reg(reg.lps,
                [cim_type, act_type, bias_type, bias_shift, quant_type, main_shift, act_shift, reduction_type],
                force = True
                )
        if bias_type == 0 and reduction_type == 0:
            set_lps += self.set_reg(reg.bi, [0, 0])
        return set_lps

    def _register_fixed_params(self, layer_info):
        setup_code = []
        setup_code += self._register_layer_offsets(layer_info)
        if isinstance(self.main_op, ConvPoolWrapper):
            if self.input_tensor.is_vt:
                if self.main_op.dilation > 1:
                    raise ValueError("Dilated conv with virtualized Tensor is meaningless... MIDAP does not support, in {}".format(layer_info))
                if not self.input_tensor.offset == (0, 0, 0):
                    raise ValueError("Currently do not support offset-driven input tensor virtualization.. in {}".format(layer_info))
                if not self.input_tensor.scale[-1] == 1:
                    raise ValueError("Currently do not support z-axis input tensor virtualization.. in {}".format(layer_info))
                if not self.input_tensor.scale[0] == self.input_tensor.scale[1]:
                    raise ValueError("Currently do not support input tensor virtualization with different scale in {}".format(layer_info))
        setup_code += self._register_write_params()
        return setup_code

    def _register_layer_offsets(self, layer_info):
        main_op = self.main_op
        control_info = layer_info.control_info
        address_unit = self.svar.read_address_unit
        read_unit = self.svar.read_unit
        in_w, in_h, in_c = self.input_tensor.shape
        _, real_h, real_c = self.input_tensor.orig_shape
        yz_plane_size = real_h * real_c
        yz_unit = yz_plane_size // address_unit
        z_unit = real_c // address_unit
        row_unit = read_unit // address_unit # wko, fko, 8bits
        fzo = 0 if isinstance(main_op, ConvWrapper) else row_unit # 8bits
        wzo = (
            control_info.wmem_info.filter_size // address_unit
            if isinstance(main_op, ConvPoolWrapper)
            else row_unit
        ) # 16KiB // 16 = 1KiB : 10bits
        fyo = z_unit
        wyo, wio, wjo, fio, fjo = 0, 0, 0, 0, 0
        if isinstance(main_op, ConvPoolWrapper):
            dilation = main_op.dilation
            k_h, k_w = main_op.k_h, main_op.k_w
            s = main_op.stride
            in_y_scale = self.input_tensor.scale[1]
            if s > in_y_scale and s % in_y_scale != 0:
                raise ValueError("Stride must be less than or equal to, or a multiple of the upsampling scale of the input tensor")
            # Following variables should be registered
            fyo = z_unit * math.ceil(s / in_y_scale) # 16bits in maximum
            fio = yz_plane_size * dilation // address_unit # 256KiB // 16 = 16KiB: 14bits
            fjo = real_c * dilation // address_unit # 14bits
            wio = (
                math.ceil(in_c * k_h / read_unit) * (read_unit // address_unit)
                if isinstance(main_op, ConvWrapper)
                else (k_h * self.system_width // read_unit) * (read_unit // address_unit)
            )# 12bits
            wjo = (
                math.ceil(in_c / read_unit) * read_unit // address_unit
                if isinstance(main_op, ConvWrapper)
                else (self.system_width // read_unit) * (read_unit // address_unit)
            ) # 12bits
            ncr = in_c // read_unit if isinstance(main_op, ConvWrapper) else 1 # total iterations for z-axis
            if isinstance(main_op, PoolWrapper):
                wio, wjo, wzo = 0, 0, 0
            self.params = [fyo, fio, wio, wjo, z_unit, ncr]
        if isinstance(main_op, ArithmeticWrapper) and not main_op.broadcast:
            wyo = z_unit
        elif isinstance(main_op, SumWrapper):
            wio = row_unit
            wzo = 0
        elif isinstance(main_op, UpBilinearWrapper):
            wzo = 0
        ## End
        sx, sy = 1, 1
        if self.input_tensor.is_valid_type:
            sx, sy = self.input_tensor.scale[:2]
        register_code = self.set_reg(reg.rso, [0, 0], force=True)
        register_code += self.set_reg(reg.ro1, [fyo, wyo], force=True)
        register_code += self.set_reg(reg.ro2, [fzo, wzo], force=True)
        register_code += self.set_reg(reg.ro3, [fio, wio * sx], force=True)
        register_code += self.set_reg(reg.ro4, [fjo, wjo * sy, row_unit, row_unit], force=True)
        register_code += self.set_reg(reg.cyzi.no, 0, force = True)
        register_code += self.set_reg(reg.tvi1.no, 0, force = True)
        register_code += self.set_reg(reg.tvi2.no, 0, force = True)
        register_code += self.set_reg(reg.rpad.no, 0, force = True)
        return register_code

    def _register_write_params(self, main = True):
        ot = self.main_output if main else self.reduction_output
        cc = self.concurrency if main else self.num_wmem
        wo_z = cc >> self.svar.write_bits
        fwo_y = ot.orig_shape[-1] >> self.svar.write_bits
        sy = 1
        if self.input_tensor.is_valid_type:
            sy = self.input_tensor.scale[1]
        self.write_pivot_info = [0, 0, 0]
        self.write_offset_info = [fwo_y, 0, wo_z]
        return self.set_reg(reg.wo1, [fwo_y * sy, wo_z], force = True) + self.set_reg(reg.wo2, 0, force = True)
    
    @property
    def yo(self):
        return self.params[0]
    
    @property
    def fio(self):
        return self.params[1]
    
    @property
    def wio(self):
        return self.params[2]

    @property
    def wjo(self):
        return self.params[3]

    @property
    def z_unit(self):
        return self.params[4]

    @property
    def ncr(self):
        return self.params[5]

    def set_finish_generator(self):
        self.generator = self.finish_generator()

    def set_next(self, last):
        code, last_group, filter_idx = self.memory_op_generator.set_next(last, self.z_iter)
        filters_left = self.memory_op_generator.num_filters - filter_idx
        self.z_iter = min(filters_left, self.in_z_tile)
        return code, last_group, filter_idx * self.concurrency

    def generate(self, code, last_filter=False):
        running_info = [self.output_loc[0], last_filter]
        return (code, running_info)
    
    def default_generator(self, **kwargs):
        yield self.generate([])
        self.generator = self.default_generator()
    
    def default_worker(self, **kwargs):
        yield []
    
    def gen_z_mem_code(self, last):
        if isinstance(self.main_op, ArithmeticWrapper):
            return [], True, 0
        else:
            return self.set_next(last)

    def gen_x_mem_code(self, last, filter_idx):
        if isinstance(self.main_op, ArithmeticWrapper):
            ret = self.set_next(last)[0]
            self.z_iter = min(self.in_z_tile, (self.shape[-1] - filter_idx) // self.system_width)
            return ret
        else:
            return []

    def common_generator(self, head_x, tail_x, write_info, last):
        last_filter = False
        write_type = 0
        head_x_out, tail_x_out = head_x, tail_x
        if isinstance(self.main_op, ConvPoolWrapper):
            head_x_out = (head_x + self.main_op.pad_l) // self.main_op.stride
            tail_x_out = (tail_x + self.main_op.pad_l) // self.main_op.stride
        last_x_offset = self.main_output.get_output_loc(
            (head_x_out if self.main_output.reverse_write else tail_x_out, 0, 0))[0][0] - self.main_output.offset[0] + 1
        if write_info is not None:
            write_type = write_info.type
            two_y = write_info.shape[-1] >> self.svar.write_bits
            sy = 1
            if self.input_tensor.is_valid_type:
                sy = self.input_tensor.scale[1]
            code = self.set_reg(reg.wo2, two_y * sy)
            self.write_offset_info[1] = two_y
            yield self.generate(code)
        else:
            self.write_pivot_info[1] = 0
        pad_x_offset = 0
        input_tensor = self.input_tensor
        if all([
            input_tensor.is_valid_type,
            input_tensor.scale[0] > 1,
            input_tensor.flip_x
        ]):
            pad_x_offset = input_tensor.scale[0] - 1
            head_x -= pad_x_offset
            tail_x -= pad_x_offset
        while not last_filter:
            out_z_filter_idx = 0
            for idx in range(self.out_z_tile):
                mem_code, last_filter, filter_idx = self.gen_z_mem_code(last) # load_filters : wrapper
                if self.main_op.bias is not None or self.reduction_op is not None:
                    bias_pivot = self.memory_op_generator.get_bmem_addr(filter_idx)
                    bias_offset = self.concurrency // self.svar.write_unit
                    mem_code += self.set_reg(reg.bi, [bias_pivot, bias_offset])
                yield self.generate(mem_code)
                if idx == 0:
                    out_z_filter_idx = filter_idx
                self.logger.debug("Run: {}, {} , {}, {}, {}".format(head_x, tail_x, filter_idx, self.z_iter, last))
                s = self.main_op.stride if isinstance(self.main_op, ConvPoolWrapper) else 1
                pad_l = self.main_op.pad_l if isinstance(self.main_op, ConvPoolWrapper) else 0
                pad_t = self.main_op.pad_t if isinstance(self.main_op, ConvPoolWrapper) else 0
                y_offset = s * write_info.shape[-2] if write_info is not None else self.tail_y - self.head_y + 1
                pad_l += pad_x_offset
                for x in range(head_x, tail_x + 1, s):
                    self.output_loc = ((x + pad_l) // s, 0, filter_idx)
                    x_offset = self.main_output.get_output_loc(self.output_loc)[0][0] - self.main_output.offset[0]
                    for y in range(self.head_y, self.tail_y + 1, y_offset):
                        mem_code = self.gen_x_mem_code(last and x == tail_x and y + y_offset > self.tail_y, filter_idx)
                        self.output_loc = ((x + pad_l) // s, (y + pad_t) // s, filter_idx)
                        wtflag = self.save_write_pivot_info(write_info, filter_idx - out_z_filter_idx)
                        # in_x, _, _ = self.input_tensor.get_loc((x, 0, 0))
                        code = self.gen_dedicated_code(x, filter_idx, self.z_iter, head_y=y, tail_y=min(self.tail_y, y + y_offset - 1))
                        transfer_code = []
                        if all([write_type in [1, 2], wtflag, write_info is not None and (
                                (x_offset - write_info.write_crit + int(not self.main_output.reverse_write)) %
                                write_info.write_shape[0] == 0
                                or (x_offset <= write_info.write_crit and self.main_output.reverse_write)
                                or x > tail_x - s)]):
                            if self.main_output.reverse_write:
                                write_shape_x = write_info.write_shape[0] - (write_info.write_crit - last_x_offset) % write_info.write_shape[0] \
                                    if last_x_offset - x_offset < write_info.write_shape[0] else write_info.write_shape[0]
                            else:
                                write_shape_x = write_info.write_shape[0] - (write_info.write_crit - x_offset - 1) % write_info.write_shape[0]
                            write_shape_y = min(write_info.write_shape[1], self.main_output.shape[-2] - self.output_loc[1])
                            tmp_write_info = copy.copy(write_info)
                            tmp_write_info.write_shape = (write_shape_x, write_shape_y, write_info.write_shape[2])
                            transfer_code = self.gen_transfer_code(
                                tmp_write_info,
                                self.output_loc[0] if self.main_output.reverse_write else self.output_loc[0] + 1 - write_shape_x,
                                self.output_loc[1],
                                filter_idx)
                        #yield self.generate(write_code + code, last_filter)
                        if self.config.DRAM.COMM_TYPE == 'TEST_3D':
                            yield self.generate(mem_code + transfer_code + code, last_filter)
                        else:
                            yield self.generate(mem_code + code + transfer_code, last_filter)
                if last_filter:
                    break
            if write_type == 3:
                transfer_code = self.gen_transfer_code(write_info, 0, 0, out_z_filter_idx)
                yield self.generate(transfer_code, last_filter)
        self.generator = self.default_generator()

    def gen_transfer_code(self, write_info, out_x, out_y, filter_idx, main = True):
        #self.logger.info("Make Transfer Info.. out_x = {}, write_info {}".format(out_x, write_info))
        transfer_info = self._make_transfer_info(write_info, out_x, out_y, filter_idx, main = main)
        transfer_code = []
        num_transfer_info = 1
        transfer_info_offset = 0
        if self.config.DRAM.COMM_TYPE == 'TEST_3D':
            transfer_info['num_plane'] = write_info.write_shape[0]
            transfer_info['flip'] = self.main_output.reverse_write
            if write_info.write_type > 1:
                transfer_info['num_transfers'] //= write_info.write_shape[0]
                transfer_info['plane_offset'] = transfer_info['transfer_offset'] * transfer_info['num_transfers']
                if self.input_tensor.is_valid_type:
                    num_transfer_info = transfer_info['num_plane']
                    transfer_info['num_plane'] = self.input_tensor.scale[1]
                    transfer_info['plane_offset'] = transfer_info['transfer_offset']
                    transfer_info['transfer_offset'] *= self.input_tensor.scale[1]
                    transfer_info['num_transfers'] //= self.input_tensor.scale[1]
                    transfer_info_offset = write_info.write_shape[1] * self.main_output.orig_shape[-1] * (-1) ** int(self.main_output.reverse_write)
                    transfer_info['flip'] = False
            else:
                transfer_info['num_transfers'] = 1
                transfer_info['transfer_unit_size'] //= write_info.write_shape[0]
                transfer_info['transfer_offset'] //= write_info.write_shape[0]
                transfer_info['plane_offset'] = transfer_info['transfer_offset']
            if self.main_output.reverse_write:
                transfer_info['dram_pivot_address'] += (write_info.write_shape[0] - 1) * transfer_info['num_transfers'] * transfer_info['transfer_offset']
        for i in range(num_transfer_info):
            transfer_code += self.memory_op_generator.transfer_tmem(**transfer_info)
            transfer_info['dram_pivot_address'] += transfer_info_offset
        sync_code = []
        # sync_code = self.gen_sync_code()
        return sync_code + transfer_code 

    def gen_reset_code(self):
        code = []
        # code = wait_reg_zero(reg.ffd)
        code += self.trigger_and_wait(TriggerType.RESET)
        return code

    def save_write_pivot_info(self, write_info = None, z_offset = 0, main = True):
        #return: write_tmem? write_code
        ot = self.main_output if main else self.reduction_output
        if ot is None:
            self.write_pivot_info = [0, 0, 0]
            return False
        loc = self.output_loc
        locs = ot.get_output_loc(loc)
        if len(locs) > 1:
            raise RuntimeError("Do not support multiple write data")
        # get FMEM address
        out_loc = locs[0]
        on = ot.name
        ox, oy, oz = out_loc
        fmem_addr, tmem_addr = 0, 0
        wflag = 0
        if on in self.output_mapping:
            om = self.output_mapping[on]
            fmem_idx, effective_x = self.get_write_fmem_info(ox, om)
            if fmem_idx >= 0:
                address = ot.get_address((effective_x, oy, oz))
                fmem_addr = self.memory_op_generator.get_fmem_addr(fmem_idx, address, write = True)
                wflag += 0x2
        # TMEM Address
        if write_info is not None and ox + ot.offset[0] >= write_info.write_crit:
            shape = write_info.write_shape
            ty = oy
            tx = ox - write_info.write_crit - ot.offset[0]
            if write_info.write_type in [1, 2]:
                tx %= write_info.write_shape[0]
                ty %= write_info.write_shape[1]
            address = tx * shape[1] * shape[2] + ty * shape[2] + z_offset
            tmem_addr = self.memory_op_generator.get_tmem_addr(address)
            wflag += 0x1
        self.write_pivot_info = [wflag, tmem_addr, fmem_addr]
        self.logger.debug("Update Write Pivot Info: {}".format(self.write_pivot_info))
        return (wflag & 0x1) > 0

    def finish_generator(self):
        if self.reduction_op is not None:
            ro = self.reduction_output
            on = ro.name
            reduction_write_info = None
            if self.output_mapping[on].write_on_dram_pivot == 0:
                reduction_write_info = WriteInfo(1, ro.shape[-1], ro.shape, 0)
            self.output_loc = (0, 0, 0)
            #wtflag, write_code = self.gen_write_code(reduction_write_info, main = False)
            write_code = self._register_write_params(main = False)
            wtflag = self.save_write_pivot_info(reduction_write_info, main = False)
            channel_pivot = 0
            # channel_offset = self.system_width // self.svar.write_unit
            channel_offset = self.num_wmem // self.svar.write_unit
            code = self.set_reg(reg.tvi1.no, 0)
            code += self.set_reg(reg.tvi2.no, 0)
            code += self.set_reg(reg.bi, [channel_pivot, channel_offset])
            code += self.set_reg(reg.ro2, [0, 0]) 
            code += self.gen_reduction_code()
            transfer_code = []
            if wtflag:
                transfer_code = self.gen_transfer_code(reduction_write_info, 0, 0, main = False)
            yield self.generate(write_code + code + transfer_code, False)
        code = self.set_reg(reg.wi, [0, 0, 0])
        code += wait_fin(rule.PIPELINE_DEPTH)
        yield self.generate(code, False)

    def default_generator(self, **kwargs):
        # self.logger.debug("default_generator is called")
        yield self.generate([], False)
        self.generator = self.default_generator()

    def default_worker(self, **kwargs):
        # self.logger.debug("default_worker is called")
        yield []

    def gen_conv_z_code(self, in_x, *args, **kwargs):
        self.logger.warning("gen_conv_z_code is not implemented yet")
        return []
    
    def gen_conv_z_tv_code(self, in_x, *args, **kwargs):
        self.logger.warning("gen_conv_z_tv_code is not implemented yet")
        return []

    def gen_conv_yz_code(self, in_x, *args, **kwargs):
        self.logger.warning("gen_conv_yz_code is not implemented yet")
        yield []
    
    def gen_arithmetic_code(self, x, *args, **kwargs):  # generate dataflow
        self.logger.warning("gen_arithmetic_code is not implemented yet")
        return []
    
    def gen_arithmetic_tv_code(self, x, y, *args, **kwargs):  # generate dataflow
        self.logger.warning("gen_arithmetic_tv_code is not implemented yet")
        return []

    def gen_reduction_code(self):
        self.logger.warning("gen_reduction_code is not implemented yet")
        return []

