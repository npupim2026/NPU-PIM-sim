from __future__ import absolute_import, division, print_function, unicode_literals
from data_structure.virtual_tensor import VInputTensor, VOutputTensor
from data_structure.instruction_components import SLayerInfo

import logging
import logging.config

import numpy as np
from copy import copy

from config import cfg
from midap_backend.wrapper.op_wrapper import ConvPoolWrapper, ConvWrapper, DWWrapper, PoolWrapper, DWConvWrapper, ArithmeticWrapper, SumWrapper, MatMulWrapper, RoPEWrapper, TestWrapper
from midap_backend.wrapper.info import WriteInfo
from midap_simulator.dataflow import generate_dataflow_info
from midap_simulator.dma_3d_memory_manager import Descriptor3D

class RunningInfo:
    def __init__(self, x=-2, last_filter=False):
        self.x = x
        self.last_filter = last_filter

class ControlLogic:
    def __init__(self, manager):
        # Initialize System Configuration
        self.manager = manager
        self.config = manager.config
        self.memory_controller = manager.memory_controller
        self.system_width = manager.config.MIDAP.SYSTEM_WIDTH
        self.num_wmem = manager.config.MIDAP.WMEM.NUM
        self.num_fmem = manager.config.MIDAP.FMEM.NUM
        self.concurrency = self.num_wmem
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger()

    def setup(self, layer_info : SLayerInfo):
        control_info = layer_info.control_info
        self.control_info = control_info
        self.input_tensors = layer_info.input
        self.input_mapping = control_info.get_input_mapping(self.input_tensor.name)
        self.head_y = control_info.behavior_info.min_y
        self.tail_y = control_info.behavior_info.max_y
        self.num_filter_rows = control_info.wmem_info.filter_size // self.system_width
        self.in_z_tile, self.out_z_tile = control_info.behavior_info.z_tile
        self.z_iter = self.in_z_tile
        self.shape = self.input_tensor.shape
        self.modules = layer_info.modules
        self.mm = self.modules[0]
        self.main_op = self.mm.op
        self.output_shape = self.mm.output[0].orig_shape
        if isinstance(self.main_op, PoolWrapper) and self.main_op.global_pool == 1:
            self.main_op.kernel = self.shape[:-1]
        self.rm = None if len(self.modules) <= 1 else self.modules[1]
        self.reduction_op = None if self.rm is None else self.rm.op
        self.skipped_cycles = 0
        self.output_loc = (0, 0, 0)
        self.input_pivot_idx = 0  # (From) Which input fragment is on FMEM
        self.concurrency = self.num_wmem
        if any(
            [
                isinstance(self.main_op, ArithmeticWrapper),
                isinstance(self.main_op, DWWrapper),
                isinstance(self.main_op, SumWrapper),
            ]
        ):
            self.concurrency = self.system_width
        self.generator = self.default_generator()
        self.save_last_run = None
        self.loaded_fmem = (-1, -1)

    @property
    def input_tensor(self):
        return self.input_tensors[0]

    def get_input_mapping(self, tensor : VInputTensor):
        return self.control_info.get_input_mapping(tensor.name)

    def set_generator(self, head_x, tail_x, pivot_idx, write_info, last):
        self.input_pivot_idx = pivot_idx
        self.logger.info("Set Generator: from x={} to {}".format(head_x, tail_x))
        if isinstance(self.main_op, ConvPoolWrapper):
            self.generator = iter(self.convpool_generator(head_x, tail_x, write_info, last))
        elif isinstance(self.main_op, ArithmeticWrapper):
            self.generator = iter(self.arithmetic_generator(head_x, tail_x, write_info, last))
        else:
            raise RuntimeError("Unexpected main operator: {}".format(self.main_op))

    def set_finish_generator(self):
        self.generator = self.finish_generator()

    def set_next(self, last):
        if self.skipped_cycles > 0:
            self.manager.stats.increase_cycle(self.skipped_cycles)
            if self.config.MIDAP.CORE_ID >= 0:
                self.memory_controller.elapse_cycle()
            self.skipped_cycles = 0
        last_group, filter_idx = self.memory_controller.set_next(last, self.z_iter)
        filters_left = self.memory_controller.num_filters - filter_idx
        self.z_iter = min(filters_left, self.in_z_tile)
        return last_group, filter_idx * self.concurrency

    def generate(self, dataflow, last_filter=False):
        running_info = RunningInfo(x=self.output_loc[0], last_filter=last_filter)
        simulated_cycle = 1 + self.skipped_cycles
        self.skipped_cycles = 0
        self.loaded_fmem = (dataflow.fmem_idx, dataflow.fmem_row)
        return (dataflow, [running_info, simulated_cycle])

    def convpool_generator(self, head_x, tail_x, write_info, last):
        # generate (dataflow, [running_info, simulated_cycle]) #
        main_op = self.main_op
        s = main_op.stride
        last_filter = False
        while not last_filter:
            for _ in range(self.out_z_tile):
                last_filter, filter_idx = self.set_next(last)
                for x in range(head_x, tail_x + 1, s):
                    for y in range(self.head_y, self.tail_y + 1, s):
                        for filter_offset in range(self.z_iter):
                            worker = self.convpool_worker(x, y, filter_idx, filter_offset)
                            for dataflow in worker:
                                yield self.generate(dataflow, last_filter)
                if last_filter:
                    break
        self.generator = self.default_generator()

    def arithmetic_generator(self, head_x, tail_x, write_info, last):
        # generate (dataflow, [running_info, simulated_cycle]) #
        for x in range(head_x, tail_x + 1):
            _, _ = self.set_next(last and x == tail_x)
            total_iters = self.shape[-1] // self.system_width
            for z_pivot in range(0, total_iters, self.in_z_tile):
                z_iter = min(self.in_z_tile, total_iters - z_pivot)
                for dataflow in self.arithmetic_worker(x, z_pivot, z_iter):
                    yield self.generate(dataflow, True)
        self.generator = self.default_generator()

    def finish_generator(self):
        if self.reduction_op is not None:
            for dataflow in self.reduction_worker():
                yield self.generate(dataflow, False)
        dataflow = generate_dataflow_info(phase=3)
        yield self.generate(dataflow, False)
        while True:
            dataflow = generate_dataflow_info()
            yield self.generate(dataflow, False)

    def convpool_worker(self, x, y, filter_idx, filter_offset):  # generate dataflow
        # Generate Output Location
        main_op = self.main_op
        pivot_x = x + main_op.pad_l
        pivot_y = y + main_op.pad_t
        s = main_op.stride
        # if main_op.dilation > 1:
        #     self.logger.debug(f"ConvPoolWorker Call: {x, y, filter_idx, filter_offset} / dilation: {main_op.dilation}")
        if pivot_x % s != 0 or pivot_y % s != 0:
            raise ValueError(
                "Wrong conv location: ({}, {}) for main_op : {}".format(x, y, main_op)
            )
        out_x, out_y = (pivot_x // s, pivot_y // s)
        self.output_loc = (out_x, out_y, filter_idx + filter_offset * self.concurrency)
        # if main_op.dilation > 1:
        #     self.logger.debug(f"Output Loc: {self.output_loc}")
        worker = self.default_worker
        if isinstance(main_op, MatMulWrapper):
            worker = self.matmul_worker
        elif isinstance(main_op, DWWrapper):
            worker = self.depthwise_worker
        elif isinstance(main_op, ConvWrapper):
            worker = (
                self.conv_yz_worker
                if self.input_tensor.mapping_type == "default" and self.input_tensor.shape[-1] % self.system_width != 0
                else self.conv_z_worker
            )
        for dataflow in self.working(worker(x, y, filter_idx, filter_offset)):
            yield dataflow

    def working(self, worker):
        for dataflow in worker:
            yield dataflow

    def get_fmem_info(self, x, tensor = None):
        if tensor is None:
            input_mapping = self.input_mapping
        else:
            input_mapping = self.get_input_mapping(tensor)
        for fmem_idx, head, tail in input_mapping[self.input_pivot_idx :]:
            if head <= x and x < tail:
                return fmem_idx, x - head

    def sync(self):
        self.manager.stats.increase_cycle(self.skipped_cycles)
        self.skipped_cycles = 0
        if self.config.MIDAP.CORE_ID >= 0 and self.config.DRAM.COMM_TYPE == "DMA":
            self.memory_controller.memory_manager.elapse_cycle()

    def default_generator(self, **kwargs):
        yield self.generate(generate_dataflow_info())
        self.generator = self.default_generator()

    def default_worker(self, *args, **kwargs):
        # self.logger.debug("default_worker is called")
        yield generate_dataflow_info()

    def conv_z_worker(self, in_x, in_y, *args, **kwargs):
        self.logger.warning("conv_z_worker is not implemented yet")
        yield generate_dataflow_info()

    def conv_yz_worker(self, in_x, in_y, *args, **kwargs):
        self.logger.warning("conv_yz_worker is not implemented yet")
        yield generate_dataflow_info()

    def depthwise_worker(self, in_x, in_y, filter_idx):
        self.logger.warning("depthwise_worker is not implemented yet")
        yield generate_dataflow_info()

    def arithmetic_worker(self, x, *args, **kwargs):  # generate dataflow
        self.logger.warning("arithmetic_worker is not implemented yet")
        yield generate_dataflow_info()

    def weighted_sum_worker(self, x, *args, **kwargs):  # generate dataflow
        self.logger.warning("weighted_sum_worker is not implemented yet")
        yield generate_dataflow_info()

    def reduction_worker(self):
        self.logger.warning("reduction_worker is not implemented yet")
        yield generate_dataflow_info()

class ControlLogicBase(ControlLogic): # tmem aware
    def setup(self, layer_info : SLayerInfo):
        super().setup(layer_info)
        # Setup Output Info
        self.output_mapping = layer_info.control_info.fmem_info.output_mapping
        mm = layer_info.modules[0]
        self.main_output = None
        if len(mm.output) == 1:
            self.main_output = mm.output[0]
        self.reduction_output = None
        if len(layer_info.modules) > 1:
            ro = layer_info.modules[1].output
            if len(ro) == 0:
                pass
            else:
                self.reduction_output = ro[0]
        # Setup Sub Generator
        self.dedicated_generator = self.default_generator
        if isinstance(self.main_op, TestWrapper):
            return
        if isinstance(self.main_op, RoPEWrapper):
            self.dedicated_generator = self.rope_generator
        elif isinstance(self.main_op, ConvPoolWrapper):
            self.dedicated_generator = self.convpool_generator
        elif isinstance(self.main_op, ArithmeticWrapper):
            self.dedicated_generator = self.arithmetic_generator
        elif isinstance(self.main_op, SumWrapper):
            self.dedicated_generator = self.weighted_sum_generator
        else:
            raise RuntimeError("Unexpected main operator: {}".format(self.main_op))
    
    def set_generator(self, head_x, tail_x, pivot_idx, write_info, last):
        self.input_pivot_idx = pivot_idx
        self.generator = iter(self.common_generator(head_x, tail_x, write_info, last))

    def common_generator(self, head_x, tail_x, write_info, last):
        last_filter = False
        write_type = 0 if write_info is None else write_info.type
        head_x_out, tail_x_out = head_x, tail_x
        if isinstance(self.main_op, ConvPoolWrapper):
            head_x_out = (head_x + self.main_op.pad_l) // self.main_op.stride
            tail_x_out = (tail_x + self.main_op.pad_l) // self.main_op.stride
        last_x_offset = self.main_output.get_output_loc(
            (head_x_out if self.main_output.reverse_write else tail_x_out, 0, 0))[0][0] - self.main_output.offset[0] + 1
        if self.config.DRAM.COMM_TYPE == 'TEST_3D' and write_info is not None:
            wdma_descriptors = []
            x_offset_unit = -1 if self.main_output.reverse_write else 1
            x_offset = write_info.write_shape[0] * x_offset_unit
            y_offset = write_info.write_shape[1]
            head_x_out_offset = self.main_output.get_output_loc((head_x_out, 0, 0))[0][0]
            tail_x_out_offset = self.main_output.get_output_loc((tail_x_out, 0, 0))[0][0]
            crit_x = write_info.write_crit + self.main_output.offset[0]
            if self.main_output.reverse_write:
                tail_x_out_offset = max(tail_x_out_offset, crit_x)
            else:
                head_x_out_offset = max(head_x_out_offset, crit_x)
            z_size = self.memory_controller.num_filters * self.memory_controller.filter_set_size if isinstance(self.main_op, ConvWrapper) else self.main_output.shape[-1]
            aoff = self.main_output.get_address(self.main_output.offset)
            if self.main_output.mapping_type == 'wmem':
                if head_x == 0: # only one descriptor
                    for oz in range(0, z_size, write_info.write_shape[-1]):
                        if write_type == 1 or self.main_output.shape[-2] == 1:
                            wmem_x_offset = self.main_output.orig_shape[-3] // self.num_wmem
                            dt, address = self.memory_controller.memory_manager.dram_address_dict[self.main_output.name]
                            base_addr = self.memory_controller.memory_manager.dram_offset[dt] + address * self.memory_controller.memory_manager.data_size
                            head_pivot = 0
                            mid_pivot = self.main_output.shape[-3] // self.num_wmem - self.main_output.reverse_write * (1 - wmem_x_offset * (self.num_wmem - 1))
                            tail_pivot = mid_pivot + (self.main_output.shape[-3] % self.num_wmem) * wmem_x_offset
                            head_addr = self.main_output.get_address((head_pivot, 0, oz)) * self.memory_controller.memory_manager.data_size + base_addr
                            mid_addr = self.main_output.get_address((mid_pivot, 0, oz)) * self.memory_controller.memory_manager.data_size + base_addr
                            tail_addr = self.main_output.get_address((tail_pivot, 0, oz)) * self.memory_controller.memory_manager.data_size + base_addr
                            write_unit_size = write_info.write_shape[-2] * write_info.write_shape[-1]
                            offset_unit = np.prod(self.main_output.orig_shape[-2:])
                            main_desc = Descriptor3D(
                                aoff, write_unit_size, offset_unit * wmem_x_offset, self.num_wmem, offset_unit, self.main_output.shape[-3] // self.num_wmem,
                                bcsync=True, intenb=True, baddr_ddr=(mid_addr if self.main_output.reverse_write else head_addr),
                                bsub=self.main_output.reverse_write, csub=self.main_output.reverse_write, last=False
                            )
                            sub_desc = Descriptor3D(
                                aoff, write_unit_size, offset_unit * wmem_x_offset, self.main_output.shape[-3] % self.num_wmem, offset_unit, 1,
                                bcsync=True, intenb=True, baddr_ddr=(tail_addr if self.main_output.reverse_write else mid_addr),
                                bsub=self.main_output.reverse_write, csub=self.main_output.reverse_write, last=False
                            )
                            if self.main_output.shape[-3] % self.num_wmem != 0:
                                if self.main_output.reverse_write:
                                    wdma_descriptors.append(sub_desc)
                                    wdma_descriptors.append(main_desc)
                                else:
                                    wdma_descriptors.append(main_desc)
                                    wdma_descriptors.append(sub_desc)
                            else:
                                wdma_descriptors.append(main_desc)
            else:
                for oz in range(0, z_size, write_info.write_shape[-1]):
                    for ox in range(head_x_out_offset, tail_x_out_offset + x_offset_unit, x_offset):
                        for oy in range(0, self.main_output.shape[-2], y_offset):
                            dram_pivot_address = self.main_output.get_address((ox, oy, oz)) * self.memory_controller.memory_manager.data_size
                            dt, address = self.memory_controller.memory_manager.dram_address_dict[self.main_output.name]
                            dram_pivot_address += self.memory_controller.memory_manager.dram_offset[dt] + address * self.memory_controller.memory_manager.data_size
                            write_shape_x = min((tail_x_out_offset - ox) * x_offset_unit + 1, write_info.write_shape[0])
                            write_shape_y = min(self.main_output.shape[-2] - oy, y_offset)
                            if write_type == 1:
                                write_unit_size = write_shape_y * write_info.write_shape[-1]
                                descriptor = Descriptor3D(
                                    aoff, write_unit_size, write_unit_size, 1, write_unit_size, write_shape_x,
                                    bcsync=True, intenb=True, baddr_ddr=dram_pivot_address, csub=self.main_output.reverse_write, last=False
                                )
                            elif write_type == 2:
                                write_unit_size = write_info.write_shape[-1]
                                descriptor = Descriptor3D(
                                    aoff, write_unit_size, self.main_output.orig_shape[-1], write_shape_y, np.prod(self.main_output.orig_shape[1:]), write_shape_x,
                                    bcsync=True, intenb=True, baddr_ddr=dram_pivot_address, csub=self.main_output.reverse_write, last=False
                                )
                            else:
                                raise ValueError("Unexpected write type")
                            wdma_descriptors.append(descriptor)
            if wdma_descriptors:
                wdma_descriptors[-1].last = True
                self.memory_controller.memory_manager.register_wdma_descriptor(wdma_descriptors)
        while not last_filter:
            out_z_filter_idx = 0
            for idx in range(self.out_z_tile):
                last_filter, filter_idx = self.load_filters(last) # load_filters : wrapper
                if idx == 0:
                    out_z_filter_idx = filter_idx
                generator = self.dedicated_generator(head_x, tail_x, filter_idx, self.z_iter, last)
                prev_out_x = -1
                prev_out_y = -1
                prev_pivot_y = 0
                wtflag = False # Write TMEM Flag
                for dataflow in generator:
                    out_x = dataflow.out_x
                    out_y = dataflow.out_y
                    x_offset = self.main_output.get_output_loc((out_x, 0, 0))[0][0] - self.main_output.offset[0]
                    if all([
                        write_type in [1, 2],
                        prev_out_x != -1,
                        prev_out_x != out_x or (prev_out_y != out_y and write_info is not None and out_y % write_info.write_shape[1] == 0),
                        wtflag,
                        write_info is not None and (
                            (not self.main_output.reverse_write
                                and (x_offset - write_info.write_crit) % write_info.write_shape[0] == 0)
                            or (self.main_output.reverse_write
                                and (x_offset - write_info.write_crit + 1) % write_info.write_shape[0] == 0)
                            or x_offset < write_info.write_crit)
                        ]):
                        new_write_info = write_info
                        if self.main_output.reverse_write:
                            new_write_shape_x = min(write_info.write_shape[0], last_x_offset - prev_x_offset)
                            if new_write_shape_x != 0 and new_write_shape_x != write_info.write_shape[0]:
                                new_write_info = copy(write_info)
                                new_write_info.write_shape = (new_write_shape_x,
                                                              write_info.write_shape[1], write_info.write_shape[2])
                        if prev_out_x != out_x and self.main_output.shape[1] % write_info.write_shape[1] != 0:
                            new_write_shape_y = self.main_output.shape[1] - prev_pivot_y
                            new_write_info = copy(write_info)
                            new_write_info.write_shape = (write_info.write_shape[0], new_write_shape_y, write_info.write_shape[2])
                        transfer_dataflow = self.make_transfer_dataflow(
                            new_write_info,
                            prev_out_x if self.main_output.reverse_write
                            else prev_out_x + 1 - new_write_info.write_shape[0],
                            prev_pivot_y,
                            filter_idx)
                        prev_pivot_y = out_y
                        yield self.generate(transfer_dataflow, last_filter)
                    prev_out_x = out_x
                    prev_out_y = out_y
                    prev_x_offset = x_offset
                    if dataflow.last:
                        wtflag = self.merge_write_info(dataflow, out_z_filter_idx, write_info)
                    yield self.generate(dataflow, last_filter)
                if write_type in [1, 2] and wtflag:
                    new_write_info = write_info
                    new_write_shape_x = min(write_info.write_shape[0], last_x_offset - prev_x_offset) \
                        if self.main_output.reverse_write \
                        else (prev_x_offset + 1 - write_info.write_crit) % write_info.write_shape[0]
                    if new_write_shape_x != 0 and new_write_shape_x != write_info.write_shape[0]:
                        new_write_info = copy(write_info)
                        new_write_info.write_shape = (new_write_shape_x,
                                                      write_info.write_shape[1], write_info.write_shape[2])
                    if self.main_output.shape[1] % write_info.write_shape[1] != 0:
                        new_write_shape_y = self.main_output.shape[1] - prev_pivot_y
                        new_write_info = copy(write_info)
                        new_write_info.write_shape = (write_info.write_shape[0], new_write_shape_y, write_info.write_shape[2])
                    transfer_dataflow = self.make_transfer_dataflow(
                        new_write_info,
                        prev_out_x if self.main_output.reverse_write
                        else prev_out_x + 1 - new_write_info.write_shape[0],
                        prev_pivot_y,
                        filter_idx)
                    yield self.generate(transfer_dataflow, last_filter)
                if last_filter:
                    break
            if write_type == 3:
                transfer_dataflow = self.make_transfer_dataflow(write_info, 0, 0, out_z_filter_idx)
                yield self.generate(transfer_dataflow, last_filter)
        self.generator = self.default_generator()
    
    def make_transfer_dataflow(self, write_info, out_x, out_y, filter_idx, main = True):
        transfer_info = self._make_transfer_info(write_info, out_x, out_y, filter_idx, main = main)
        dataflow = generate_dataflow_info(phase=4)
        dataflow.transfer_info = transfer_info
        return dataflow
    
    def _make_transfer_info(self, write_info, out_x, out_y, filter_idx, main):
        if write_info is None or write_info.type == 0:
            raise RuntimeError("make_transfer_dataflow should not be called")
        ot : VOutputTensor = self.main_output if main else self.reduction_output
        if ot is None:
            raise RuntimeError("make_transfer_dataflow should not be called")
        tu, nt, to = 0, 0, 0
        if write_info.type == 1:
            tu = np.prod(write_info.shape)
            nt = 1
            to = tu
        else:
            tu = write_info.shape[-1]
            nt = write_info.shape[0] * write_info.shape[1]
            to = ot.orig_shape[2]
        tpa = 0 # Can be modified in future..
        loc = (out_x, out_y, filter_idx)
        out_loc = ot.get_output_loc(loc)[0]
        crit_x, oy, oz = out_loc
        if write_info.type == 3:
            crit_x = write_info.crit + ot.offset[0]
        elif crit_x < write_info.crit + ot.offset[0]:
            raise RuntimeError("crit_x < write_info.crit")
        if ot.mapping_type == 'wmem':
            crit_x = (crit_x % self.num_wmem) * (ot.orig_shape[0] // self.num_wmem) + crit_x // self.num_wmem
        dpa = ot.get_address((crit_x, oy, oz))
        transfer_info = dict(
            data_name = ot.name,
            dram_pivot_address = dpa,
            transfer_unit_size = tu,
            transfer_offset = to,
            num_transfers = nt,
            tmem_pivot_address = tpa,
            )
        return transfer_info

    def merge_write_info(self, dataflow, filter_idx, write_info = None, main = True):
        # FMEM Address
        ot : VOutputTensor = self.main_output if main else self.reduction_output
        if ot is None:
            return False
        loc = dataflow.out_loc
        locs = ot.get_output_loc(loc)
        if len(locs) > 1:
            raise RuntimeError("Do not support multiple write data")
        out_loc = locs[0]
        # For functinality checking, reserve this info
        dataflow.out_loc = out_loc
        # get FMEM address
        on = ot.name
        ox, oy, oz = out_loc
        if on in self.output_mapping:
            om = self.output_mapping[on]
            fmem_idx, effective_x = self.get_write_fmem_info(ox, om)
            if fmem_idx >= 0:
                address = ot.get_address((effective_x, oy, oz))
                dataflow.write_fmem_addr = (fmem_idx, address)
        # TMEM Address
        if write_info is None:
            return False
        if write_info.write_crit > ox + ot.offset[0]:
            return False
        shape = write_info.write_shape
        ty = oy
        tx = ox - write_info.write_crit - ot.offset[0]
        if write_info.write_type in [1, 2]:
            tx %= write_info.write_shape[0]
            ty %= write_info.write_shape[1]
        z_offset = loc[-1] - filter_idx
        tmem_addr = tx * shape[1] * shape[2] + ty * shape[2] + z_offset
        dataflow.write_tmem_addr = tmem_addr
        return True
    
    def get_write_fmem_info(self, x, mapping):
        fmem_idx = -1
        for idx, head, tail in mapping:
            if head <= x and x < tail:
                fmem_idx = idx
                effective_x = x - head
                return fmem_idx, effective_x
        return -1, -1

    def load_filters(self, last):
        if isinstance(self.main_op, ArithmeticWrapper):
            return True, 0
        else:
            return self.set_next(last)
    
    def convpool_generator(self, head_x, tail_x, filter_idx, z_iter, *args, **kwargs):
        # generate (dataflow, [running_info, simulated_cycle]) #
        main_op = self.main_op
        s = main_op.stride
        for x in range(head_x, tail_x + 1, s):
            for y in range(self.head_y, self.tail_y + 1, s):
                for filter_offset in range(z_iter):
                    worker = self.convpool_worker(x, y, filter_idx, filter_offset)
                    for dataflow in worker:
                        yield dataflow

    def arithmetic_generator(self, head_x, tail_x, filter_idx, z_iter, last, *args, **kwargs):
        # generate (dataflow, [running_info, simulated_cycle]) #
        for x in range(head_x, tail_x + 1):
            _, _ = self.set_next(last and x == tail_x)
            total_iters = self.shape[-1] // self.system_width
            for z_pivot in range(0, total_iters, self.in_z_tile):
                z_iter = min(self.in_z_tile, total_iters - z_pivot)
                for dataflow in self.arithmetic_worker(x, z_pivot, z_iter):
                    yield dataflow
    
    def weighted_sum_generator(self, head_x, tail_x, filter_idx, z_iter, last, *args, **kwargs): #FIXME # Temporal Implementation
        # generate (dataflow, [running_info, simulated_cycle]) #
        total_iters = self.shape[-1] // self.system_width
        start_iter = filter_idx // self.system_width
        z_iter = min(self.in_z_tile, total_iters - start_iter)
        for x in range(head_x, tail_x + 1):
            for dataflow in self.weighted_sum_worker(x, start_iter, z_iter):
                yield dataflow

    def rope_generator(self, head_x, tail_x, filter_idx, z_iter, last, *args, **kwargs):
        # TODO: input sequence longer than 1 token length should be able to handled
        start_iter = max(head_x, filter_idx)
        last_iter = min(tail_x + 1, filter_idx + z_iter)
        z_iter = min(self.in_z_tile, last_iter - start_iter)
        for x in range(start_iter, start_iter + z_iter):
            for dataflow in self.rope_worker(x, 0, self.shape[-1] // self.system_width):
                yield dataflow
    
    def finish_generator(self):
        if self.reduction_op is not None:
            reduction_write_info = None
            ro = self.reduction_output
            on = ro.name
            if self.output_mapping[on].write_on_dram_pivot == 0:
                reduction_write_info = WriteInfo(1, ro.shape[-1], ro.shape, 0)
            for dataflow in self.reduction_worker():
                self.merge_write_info(dataflow, 0, reduction_write_info, main = False) # TODO
                yield self.generate(dataflow, False)
            if reduction_write_info is not None:
                transfer_dataflow = self.make_transfer_dataflow(reduction_write_info, 0, 0, 0)
                yield self.generate(transfer_dataflow, False)
        dataflow = generate_dataflow_info(phase=3)
        yield self.generate(dataflow, False)
        while True:
            dataflow = generate_dataflow_info()
            yield self.generate(dataflow, False)


