from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from collections import deque

import numpy as np

from config import cfg
from midap_backend.wrapper.op_wrapper import ArithmeticWrapper, DWWrapper, SumWrapper

from .stage import Stage
from .dataflow import generate_dataflow_info

class FWriteStage(Stage):
    def initialize(self):
        self.memory_controller = self.manager.memory_controller
        self.quant = self.manager.config.MODEL.QUANTIZED
        data_type = np.int8 if self.quant else np.float32
        self.concurrency = self.num_wmem
        self.valid_bit_len = self.system_width // self.num_wmem
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger("debug")
    
    @property
    def data_info_dict(self):
        return self.manager.data_info_dict

    def setup(self, modules): # microcontroller
        output_mapping = self.manager.control_info.fmem_info.output_mapping
        op = modules[0].op
        if any([isinstance(op, ArithmeticWrapper), isinstance(op, DWWrapper), isinstance(op, SumWrapper)]):
            self.concurrency = self.system_width
            self.valid = tuple(True for _ in range(self.valid_bit_len))
        else:
            self.concurrency = self.num_wmem
            self.valid = tuple(False for _ in range(self.valid_bit_len))
        cc = self.concurrency
        mm = modules[0]
        if len(mm.output) > 1:
            raise RuntimeError("Do not support multiple write data")
        self.main_output = None
        if len(mm.output) == 0:
            pass
        else:
            mo = mm.output
            if all([mo[0].offset[-1] % cc != 0 or mo[0].shape[-1] % cc != 0,
                len(output_mapping[mo[0].name]) > 0]):
                raise RuntimeError("Unaligned output: op {}, output {} cannot be processed".format(op, mo[0]))
            self.main_output = mo[0]
        self.reduction_output = None
        if len(modules) > 1:
            ro = modules[1].output
            if len(ro) > 1:
                raise RuntimeError("Do not support multiple write data")
            if len(ro) == 0:
                pass
            else:
                if all([ro[0].offset[-1] % cc != 0 or ro[0].shape[-1] % cc != 0,
                    len(output_mapping[ro[0].name]) > 0]):
                    raise RuntimeError("Unaligned output: op {}, output {} cannot be processed".format(modules[1], ro[0]))
                self.reduction_output = ro[0]

    def run(self, info):
        self.info = info
        if info.phase in [0, 3]:
            pass
        elif info.phase in [1, 2]:
            self.write(info)
        elif info.phase == 4:
            self.memory_controller.transfer_tmem(**info.transfer_info)
            self.logger.debug("Transfer_TMEM: input {}".format(info.transfer_info))
        return info

    def write(self, info):
        if not info.last:
            return
        input_buf = self.input_buf
        on, ws = 0, 0
        if info.phase == 1:
            on = self.main_output.name
            ws = self.concurrency
        elif info.phase == 2:
            on = self.reduction_output.name
            ws = self.system_width
            self.valid = tuple(True for _ in range(self.valid_bit_len))
        # For functinality checking
        ox, oy, oz = info.out_loc
        offset = oz % self.system_width
        self.write_data_info(on, ox, oy, oz, oz + ws, input_buf[offset:offset+ws])
        #self.data_info_dict[on].compare_data[x, y, head:tail] = self.save_buf[wo:wo+ws]
        fmem_idx, address = info.write_fmem_addr
        if fmem_idx >= 0:
            wo = (address % self.system_width) // self.num_wmem
            valid = list(self.valid)
            valid[wo] = True
            self.memory_controller.write_fmem(fmem_idx, address, input_buf, valid)
        tmem_addr = info.write_tmem_addr
        if tmem_addr >= 0:
            wo = (tmem_addr % self.system_width) // self.num_wmem
            valid = list(self.valid)
            valid[wo] = True
            self.memory_controller.write_tmem(tmem_addr, input_buf, valid)

    def write_data_info(self, output_name, x, y, head, tail, data):
        # self.logger.debug(f"{x}, {y}, {head}:{tail}")
        if self.data_info_dict is not None:
            self.data_info_dict[output_name].compare_data_logical[x, y, head:tail] = data[:]

class VWriteStage(FWriteStage):
    def write_data_info(self, *args, **kwargs):
        pass

