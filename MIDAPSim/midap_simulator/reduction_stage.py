from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from midap_backend.wrapper.op_wrapper import DWWrapper, AvgpoolWrapper, ArithmeticWrapper, SumWrapper

from .stage import Stage
from .dataflow import generate_dataflow_info


class FReductionStage(Stage):
    def initialize(self):
        self.quant = self.manager.config.MODEL.QUANTIZED
        data_type = np.int32 if self.quant else np.float32
        self.reduction_type = None
        self.reduction_buf = np.zeros(self.manager.config.MIDAP.REDUCTION.NUM_ENTRIES, dtype = data_type)
        self.reduction_value = 0
        self.quant_info = None
        self.quant_buf = None if not self.quant else np.zeros(self.system_width, dtype=np.int64)
        self.bypass_flag = True
        data_type = np.int8 if self.quant else np.float32
        self.output_buf = np.zeros(self.system_width, dtype = data_type)
        self.concurrency = self.num_wmem
        self.input_buf = None
        self.logger = self.manager.logger

    def setup(self, modules):
        op = modules[0].op
        self.concurrency = self.num_wmem
        if any([isinstance(op, ArithmeticWrapper), isinstance(op, DWWrapper), isinstance(op, SumWrapper)]):
            self.concurrency = self.system_width
        self.reduction_type = 0 # No reduction
        self.reduction_value = 0
        if len(modules) > 1:
            reduction_op = modules[1].op
            self.quant_info = modules[1].quant_info
        else:
            reduction_op = None
        if reduction_op is None:
            pass
        elif isinstance(reduction_op, AvgpoolWrapper):
            self.reduction_type = 1
            self.reduction_value = reduction_op.k_w * reduction_op.k_h
        else:
            raise ValueError("Unknown Reduction operation {}".format(reduction_op))

    def run(self, dataflow_info):
        info = dataflow_info
        input_buf = self.input_buf
        if self.reduction_type == 0 or info.phase in [0, 3]:
            pass
        if info.phase == 1: # Phase 1
            self.output_buf[:] = input_buf[:] # bypass
            if not info.last:
                pass
            elif self.reduction_type == 1: # update reduction buf
                filter_idx = info.channel_idx
                offset = filter_idx % self.system_width
                if info.first:
                    self.reduction_buf[filter_idx:filter_idx + self.concurrency] = input_buf[offset:offset+self.concurrency]
                else:
                    self.reduction_buf[filter_idx:filter_idx + self.concurrency] = \
                            np.add(self.reduction_buf[filter_idx:filter_idx + self.concurrency], input_buf[offset:offset+self.concurrency])
                #self.logger.debug("reduction update - loc: {}, updated data: {}".format((info.out_x, info.out_y, info.out_z), self.reduction_buf[filter_idx:filter_idx+4]))
        elif info.phase == 2: # Phase 2
            if not self.quant:
                self.output_buf[:self.system_width] = \
                        np.true_divide(self.reduction_buf[info.channel_idx:info.channel_idx + self.system_width], self.reduction_value)
            else: # Activation Logic
                n1 = self.quant_info.n1
                n2 = self.quant_info.n2
                if self.quant_info.f == 1:
                    self.quant_buf[:] = np.right_shift(
                        self.reduction_buf[info.channel_idx:info.channel_idx + self.system_width], n1
                    )
                else:
                    self.quant_buf[:] = np.left_shift(
                        self.reduction_buf[info.channel_idx:info.channel_idx + self.system_width], n1
                    )
                self.__do_truncate(16, self.quant_buf)
                self.quant_buf[:] = np.right_shift(
                    self.quant_buf, n2
                )
                self.__do_truncate(8, self.quant_buf)
                self.output_buf[:] = self.quant_buf[:]
        return info
    
    def __do_truncate(self, n, buf):
        minimum = int(-(2 ** (n-1)))
        maximum = int((2 ** (n-1)) - 1)
        if self.quant:
            buf[:] = np.where(buf > maximum, maximum, buf)
            buf[:] = np.where(buf < minimum, minimum, buf)
     
class VReductionStage(Stage):
    def initialize(self):
        self.output_buf = np.zeros(self.system_width)
    def run(self, info):
        return info
