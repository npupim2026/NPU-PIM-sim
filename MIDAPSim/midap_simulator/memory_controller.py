from __future__ import absolute_import, division, print_function, unicode_literals
from data_structure.instruction_components import SLayerInfo
from software.compiler.wmem_info import ComputeType
from software.system_compiler.memory_info import MemoryType
from typing import Any, List

import numpy as np
import logging
import copy
import os
import math

from config import cfg

from .virtual_memory_manager import VMemoryManager, TVMemoryManager
from .dma_memory_manager import DMemoryManager, TDMemoryManager
from .dma_3d_memory_manager import DMA3DMemoryManager

def get_memory_manager(manager):
    ct = manager.config.DRAM.COMM_TYPE
    if ct == 'VIRTUAL':
        return TVMemoryManager(manager)
    if ct == 'TEST_DMA':
        return TDMemoryManager(manager)
    elif ct == 'DMA':
        return DMemoryManager(manager)
    elif ct == 'TEST_3D':
        return DMA3DMemoryManager(manager)

class MemoryController():
    def __init__(self, manager):
        self.manager = manager
        self.memory_manager = get_memory_manager(manager)
        self.system_width = manager.config.MIDAP.SYSTEM_WIDTH
        self.filter_name = None
        self.bias_name = None
        self.compute_type = ComputeType.StdConv
        self.input_tensor = None
        self.num_filters = 0
        self.load_filter_once = False
        self.all_filters_on_wmem = False
        self.filter_size = 0
        self.prepare_info = None
        self.num_wmem = manager.config.MIDAP.WMEM.NUM
        self.filter_group_size = 1
        self.channel_broadcast = False
        self.next_filters_on_wmem = False
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('debug')
        self.layer_info = None

    def set_dram_info(self, dram_data, dram_address_dict):
        self.memory_manager.set_dram_info(dram_data, dram_address_dict)

    def reset_sram(self):
        self.memory_manager.reset_sram()
    
    def store(self):
        mm_store = self.memory_manager.store()
        return [self.group_idx, self.all_filters_on_wmem, mm_store]

    def restore(self, save):
        self.group_idx = save[0]
        self.all_filters_on_wmem = save[1]
        self.memory_manager.restore(save[2])

    def setup(self, layer_info : SLayerInfo):
        control_info = layer_info.control_info
        # Load fmem information
        self.input_mapping = control_info.fmem_info.input_mapping
        self.layer_info = layer_info
        # Load wmem information
        self.wmem_info = control_info.wmem_info
        wi = self.wmem_info
        self.compute_type = wi.compute_type
        # Filter information
        self.filter_name = wi.filter_name
        self.bias_name = wi.bias_name
        self.filter_size = wi.filter_size
        self.num_filters = wi.num_filters
        self.use_extended_cim = False
        if self.filter_size % self.system_width != 0:
            raise ValueError("Filter size is not padded as well")
        if self.compute_type in [ComputeType.StdConv, ComputeType.MatMulTrans]: # Conv
            self.filter_set_size = self.num_wmem
        elif self.compute_type in [ComputeType.DWConv, ComputeType.Pool, ComputeType.Elwise, ComputeType.WeightedSum, ComputeType.MatMul]: # DWConv, Pool
            self.filter_set_size = 1
            self.use_extended_cim = True
        # elif self.compute_type == ComputeType.Elwise: # ArithmeticOp
        #     self.filter_set_size = 1
        #     self.use_extended_cim = True
        self.num_filters = self.num_filters // self.filter_set_size
        # Group related terms
        self.filter_group_size = wi.filter_group_size
        self.num_filter_groups = math.ceil(self.num_filters/self.filter_group_size)
        #self.filter_idx = -1
        self.filter_offset = 0
        self.group_idx = -1
        # Filter loading configuration
        self.reverse_load = wi.reverse_load
        self.load_filter_once = wi.load_filter_once
        self.all_filters_on_wmem = self.filter_name is None
        self.channel_broadcast = wi.compute_type == ComputeType.WeightedSum # TODO: There may be more operations require this option
        # preparation information
        if self.filter_name is not None:
            if not self.wmem_info.prepared:
                self.memory_manager.reset_wmem()
                self.load_wmem(False)
                if self.bias_name is not None:
                    self.load_bmem(self.bias_name, wi.bias_size)
            elif self.next_filters_on_wmem:
                self.all_filters_on_wmem = True
                self.next_filters_on_wmem = False
        if wi.lut_name is not None:
            self.load_lut(wi.lut_name)
        self.prepare_info = wi.prepare_info
        if self.prepare_info is not None and self.load_filter_once and self.prepare_info.filter_name == self.filter_name:
            self.next_filters_on_wmem = True
        self.setup_bmem()
        self.logger.debug(f"{self}")

    def __repr__(self):
        ret = ""
        if self.layer_info is None:
            return ret
        ret += "=========================================\n"
        ret += f"Memory Controller initialization information for Layer {self.layer_info.name}\n"
        ret += "=========================================\n"
        ret += f"[Compute Type: {self.compute_type}, Filter: {self.filter_name}]\n"
        ret += f"[Num filters: {self.num_filters}, Filter Size: {self.filter_size}]\n"
        ret += f"[Filter Group Size: {self.filter_group_size}, Num Filter Groups: {self.num_filter_groups}]\n"
        ret += f"[Its WMEM is prefetched? {self.wmem_info.prepared}]\n"
        if self.prepare_info is not None:
            ret += f"It'll prefech wmem data for {self.prepare_info.filter_name}]\n"
        else:
            ret += "It'll not prefech wmem data\n"
        ret += "=========================================\n"
        return ret
   
    def time(self):
        return self.manager.stats.total_cycle()

    def sync(self, sync_id, compute_cycle, layer_id):
        self.memory_manager.sync(sync_id, compute_cycle, layer_id)
        #self.logger.debug("MemoryController @ {}: Generate sync id {}".format(self.time, sync_id))
    
    def wait_sync(self, wait_id, layer_id):
        wait_time = self.memory_manager.wait_sync(wait_id, layer_id)
        self.manager.stats.wait_sync(wait_time)
        # self.logger.info("MemoryController @ {}: Wait for sync id {}, time = {}".format(self.time, wait_id, wait_time))

    # WMEM related functions
    def set_next(self, last_use = False, z_iters = 1):
        self.logger.debug("====================================================")
        self.logger.debug(f"set_next({(last_use, z_iters)}) is called")
        self.logger.debug(f"Current Status: [foffset = {self.filter_offset}, group_idx: {self.group_idx}]")
        next_foffset = self.filter_offset + z_iters
        check = next_foffset + self.group_idx * self.filter_group_size >= self.num_filters
        load_next = self.group_idx < 0
        if self.compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum, ComputeType.MatMul, ComputeType.MatMulTrans]:
            if next_foffset >= self.filter_group_size or check:
                load_next = True
        elif self.compute_type in [ComputeType.Pool]:
            if next_foffset >= self.num_filters:
                next_foffset = 0
        else:
            load_next = True
        if load_next:
            _, _ = self._set_next(last_use)
            self.filter_offset = 0
        else:
            self.filter_offset = next_foffset
        curr_idx = self.group_idx * self.filter_group_size + self.filter_offset
        last = curr_idx + z_iters >= self.num_filters
        self.logger.debug(f"Internal Info: [load_next = {load_next}]")
        self.logger.debug(f"Updated Status: [foffset = {self.filter_offset}, group_idx: {self.group_idx}]")
        self.logger.debug(f"Return: [last = {check}, curr_idx: {curr_idx}]")
        self.logger.debug("====================================================")
        return last, curr_idx

    def _set_next(self, last_use=False):
        self.group_idx = (self.group_idx + 1) % self.num_filter_groups
        last_group = self.group_idx + 1 == self.num_filter_groups
        if self.compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum, ComputeType.MatMul, ComputeType.MatMulTrans]:
            switch_wmem = True
            if self.all_filters_on_wmem and self.group_idx == 0:
                switch_wmem = self.num_filter_groups % 2 == 0
            if switch_wmem:
                self.memory_manager.switch_wmem()
            self.load_wmem(last_use)
        elif self.compute_type == ComputeType.Elwise:
            if not self.all_filters_on_wmem or self.num_filter_groups > 1:
                self.memory_manager.switch_wmem()
            self.load_wmem(last_use)
        self.logger.debug("Time {}: WMEM IN USE: {}".format(self.time, self.memory_manager.wmem_in_use))
        filter_idx = self.group_idx * self.filter_group_size
        return last_group, filter_idx

    def load_wmem(self, last_use = False):
        #load = self.filter_idx % self.filter_group_size == 0 or self.filter_idx == -1
        #if not load:
        #    return None
        #load_filter_idx = 0 if self.filter_idx == -1 else self.filter_idx + self.filter_group_size
        load_filter_idx = (self.group_idx + 1) * self.filter_group_size
        load_prepare = False
        if load_filter_idx >= self.num_filters:
            if self.load_filter_once:
                self.all_filters_on_wmem = True
            load_prepare = last_use
            load_filter_idx = 0
        self.logger.debug("Load WMEM: Load filter idx : {} to wmem {}, load_prepare : {}, all_filters_on_wmem: {}".format(load_filter_idx, (self.memory_manager.wmem_in_use + 1) % 2,  load_prepare, self.all_filters_on_wmem))
        if not load_prepare:
            if self.all_filters_on_wmem:
                return None
            next_group_size = min(self.num_filters - load_filter_idx, self.filter_group_size) if not self.channel_broadcast else 1
            if self.compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum, ComputeType.MatMul, ComputeType.MatMulTrans]:
                wmem_pivot = 0 if not self.load_filter_once else self.filter_group_size * (load_filter_idx // (2 * self.filter_group_size))
                filter_offset = self.filter_size * next_group_size if self.compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum] else self.filter_size * self.num_filters // self.filter_set_size
                self.load_filter(
                        self.compute_type,
                        self.filter_name,
                        self.filter_size,
                        self.layer_info.name,
                        next_group_size,
                        filter_offset,
                        wmem_pivot,
                        load_filter_idx,
                        reorder_load=self.wmem_info.reorder_load,
                        )
            elif self.compute_type == ComputeType.Elwise:
                filter_idx_pivot = self.num_filters - load_filter_idx - next_group_size if self.reverse_load else load_filter_idx
                self.load_filter(
                        self.compute_type,
                        self.filter_name,
                        self.filter_size,
                        self.layer_info.name,
                        next_group_size,
                        next_group_size * self.filter_size,
                        0,
                        filter_idx_pivot,
                        self.reverse_load
                        )
        elif self.prepare_info is None or self.next_filters_on_wmem:
            return None
        else:
            self.logger.debug("Load Prepare Info")
            pi = self.prepare_info
            if pi.compute_type == ComputeType.Elwise:
                load_idx = 0 if not pi.reverse_load else pi.num_filters - pi.filter_group_size
                self.load_filter(
                        pi.compute_type,
                        pi.filter_name,
                        pi.filter_size,
                        self.layer_info.name,
                        pi.filter_group_size if pi.compute_type != ComputeType.WeightedSum else 1,  # TODO: Temporal solution
                        pi.filter_size * pi.filter_group_size,
                        0,
                        load_idx,
                        pi.reverse_load
                        )
            else:
                filter_offset = pi.filter_size * pi.filter_group_size if pi.compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum] else pi.filter_size * pi.num_filters // self.num_wmem
                self.load_filter(
                        pi.compute_type,
                        pi.filter_name,
                        pi.filter_size,
                        self.layer_info.name,
                        pi.filter_group_size if pi.compute_type != ComputeType.WeightedSum else 1,  # TODO: Temporal solution
                        filter_offset,
                        0,
                        0,
                        )

    def load_filter(
            self,
            compute_type,
            filter_name,
            filter_size,
            layer_name,
            group_size,
            filter_offset,
            wmem_pivot,
            filter_idx_pivot,
            reverse_load = False,
            reorder_load = False,
            ):
        self.logger.debug("Function call: Load Filter")
        self.logger.debug("Time: {}, compute_type: {}, filter_name: {}, filter_size: {}, \n group_size: {}, wmem_pivot: {}, filter_idx_pivot: {}, reverse_load: {}".format(self.time, compute_type, filter_name, filter_size, group_size, ((self.memory_manager.wmem_in_use + 1)%2 ,wmem_pivot), filter_idx_pivot, reverse_load))
        if compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum] or (compute_type in [ComputeType.MatMul, ComputeType.MatMulTrans] and not reorder_load):
            to_all = compute_type in [ComputeType.StdConv, ComputeType.MatMulTrans]
            num_wmem = self.num_wmem if to_all else 1
            self.memory_manager.load_wmem(
                    to_all = to_all,
                    filter_name = filter_name,
                    filter_size = filter_size * group_size,
                    filter_offset = filter_offset,
                    wmem_offset = wmem_pivot * filter_size,
                    dram_offset = filter_idx_pivot * filter_size * self.filter_set_size,
                    continuous_request = False,
                    layer_name = layer_name )
            self.manager.stats.read_dram2wmem(group_size * filter_size * num_wmem)
            self.manager.stats.write_wmem(math.ceil(group_size * filter_size/self.system_width) * num_wmem)
        elif compute_type == ComputeType.MatMul:
            if self.num_filters == 1:
                self.memory_manager.load_wmem(
                        to_all = False,
                        filter_name = filter_name,
                        filter_size = filter_size * group_size,
                        filter_offset = filter_size * group_size,
                        wmem_offset = wmem_pivot * filter_size,
                        dram_offset = filter_idx_pivot * filter_size * self.filter_set_size,
                        continuous_request = False,
                        layer_name = layer_name)
            else:
                num_rows = filter_size // self.system_width
                for i in range(num_rows):
                    self.memory_manager.load_wmem(
                        to_all = False,
                        filter_name = filter_name,
                        filter_size = self.system_width,
                        filter_offset = self.system_width,
                        wmem_offset = wmem_pivot + i * self.system_width,
                        dram_offset = (filter_idx_pivot + self.num_filters * i) * self.system_width,
                        continuous_request = False,
                        layer_name = layer_name)
        elif compute_type == ComputeType.MatMulTrans:
            for group_idx in range(group_size):
                self.memory_manager.load_wmem(
                    to_all = True,
                    filter_name = filter_name,
                    filter_size = filter_size,
                    filter_offset = filter_size,
                    wmem_offset = (wmem_pivot + group_idx) * filter_size,
                    dram_offset = (filter_idx_pivot + group_idx) * filter_size * self.num_wmem,
                    continuous_request = False,
                    layer_name = layer_name)
                self.manager.stats.read_dram2wmem(group_size * filter_size)
                self.manager.stats.write_wmem(math.ceil(group_size * filter_size/self.system_width))
        else:
            for g in range(group_size):
                if reverse_load:
                    pivot = filter_idx_pivot + group_size - g - 1
                else:
                    pivot = filter_idx_pivot + g
                self.memory_manager.load_wmem(
                        to_all = False,
                        filter_name = filter_name,
                        filter_size = filter_size,
                        filter_offset = filter_size,
                        wmem_offset = g * filter_size,
                        dram_offset = pivot * filter_size,
                        continuous_request = False,
                        layer_name = layer_name
                        )
                self.manager.stats.read_dram2wmem(filter_size)
                self.manager.stats.write_wmem(math.ceil(filter_size/self.system_width))

    def load_wbuf(self, wbuf, row):
        pivot = 0
        if self.load_filter_once:
            #pivot = (self.filter_idx // (self.filter_group_size * 2)) * self.filter_group_size
            pivot = (self.group_idx // 2) * self.filter_group_size
        if self.compute_type != ComputeType.WeightedSum:
            pivot += self.filter_offset
        #pivot += self.filter_idx % self.filter_group_size
        address = self.system_width * row + pivot * self.filter_size
        try:
            wait_time = self.memory_manager.read_wmem(wbuf, self.use_extended_cim, address)
        except Exception:
            self.logger.error(f"Error occurs while reading row {row}.")
            self.logger.error(f"Current status: pivot = {pivot}, group_idx = {self.group_idx}, filter_offset = {self.filter_offset}, filter_size = {self.filter_size}")
            raise RuntimeError

        self.manager.stats.wait_dram2wmem(wait_time)
        self.manager.stats.read_wmem()
        if wait_time > 0:
            self.logger.debug("Time {}: wmem read delay: {}".format(self.time, wait_time))

    def load_fmem(self, fmem_idx, data_name, info, layer_name):
        inp = self.input_mapping[data_name]
        data_size = (info[1] - info[0]) * inp.yz_plane_size
        data_address = inp.yz_plane_size * info[0]
        # self.logger.info("Time {}: LOAD FMEM bank {}: DATA {}, address {}, size {}".format(self.time, fmem_idx, data_name, data_address, data_size))
        self.memory_manager.load_fmem(
                fmem_idx,
                data_name,
                data_size,
                0,
                data_address,
                layer_name
                )
        self.manager.stats.read_dram2fmem(data_size)
        self.manager.stats.write_fmem(math.ceil(data_size/self.system_width))

    def load_fbuf(self, fbuf, bank_idx, row):
        address = self.system_width * row
        wait_time = self.memory_manager.read_fmem(fbuf, bank_idx, address)
        self.manager.stats.wait_dram2fmem(wait_time)
        self.manager.stats.read_fmem()
        if wait_time > 0:
            self.logger.debug("Time {}: read fmem_idx {}, delay: {}".format(self.time, bank_idx, wait_time))

    def write_fmem(self, bank_idx, address, data, valid):
        # set debug info
        self.memory_manager.write_fmem(bank_idx, address, data, valid)

    def write_tmem(self, address, data, valid): # 
        wait_time = self.memory_manager.write_tmem(address, data, valid)
        self.manager.stats.wait_write_dram(wait_time)
        self.manager.stats.write_tmem(1)
    
    def transfer_tmem(
            self,
            data_name,
            dram_pivot_address,
            transfer_unit_size,
            transfer_offset,
            num_transfers,
            tmem_pivot_address = 0,
            ):
        self.memory_manager.tmem_to_dram(
            data_name,
            dram_pivot_address,
            transfer_unit_size,
            transfer_offset,
            num_transfers,
            tmem_pivot_address = tmem_pivot_address,
            )
        self.memory_manager.switch_tmem()
        self.manager.stats.write_dram(transfer_unit_size * num_transfers)

    # Will be deprecated
    def write_dram(self, data_name, address, data, offset, size):  # DRAM Write
        wait_time = self.memory_manager.write_dram(data_name, address, data, offset, size)
        self.manager.stats.wait_memory(wait_time)
        self.manager.stats.write_dram(data.size)

    def load_bmem(self, bias_name, num_filters):
        self.memory_manager.load_bmem(bias_name, num_filters)
        self.manager.stats.read_dram(num_filters)
        self.manager.stats.write_bmem(num_filters // self.system_width)

    def setup_bmem(self):
        if self.bias_name is not None:
            self.memory_manager.switch_bmem()
        if self.prepare_info is not None and self.prepare_info.bias_name is not None:
            pi = self.prepare_info
            self.load_bmem(pi.bias_name, pi.bias_size)

    def load_bbuf(self, bbuf, address):
        wait_time = self.memory_manager.read_bmem(bbuf, address)
        self.manager.stats.wait_dram2wmem(wait_time)
        if wait_time > 0:
            self.logger.debug("Time {}: bmem read delay: {}".format(self.time, wait_time))

    def load_lut(self, lut_name):
        self.memory_manager.load_lut(lut_name)
        self.manager.stats.read_dram(self.memory_manager.lut_raw.size)

    def get_lut_items(self, x1):
        wait_time = self.memory_manager.access_lut()
        self.manager.stats.wait_dram2wmem(wait_time)
        if wait_time > 0:
            self.logger.debug("Time {}: lut read delay: {}".format(self.time, wait_time))
        lut_raw = self.memory_manager.lut[x1, :].astype(np.int16)
        lut = np.zeros([lut_raw.shape[0], 2])
        for i in range(2):
            lut[:, i] = np.left_shift(lut_raw[:, 2*i+1], 8) + lut_raw[:,2*i]
        return lut.astype(np.int32)

    def load_host(self, name, size):
        self.memory_manager.load_host(name, size)
        self.manager.stats.read_dram(size)

    def write_host(self, name, data):
        self.memory_manager.write_host(name, data)
        self.manager.stats.write_dram(data.size)

    def sync_host(self):
        wait_time = self.memory_manager.sync_host()
        self.manager.stats.wait_memory(wait_time)

    # Null function to sync with DMA
    def elapse_cycle(self):
        return

    ## Dumping
    def dump_gold_data(self, output_dir, output_only=False):
        if self.memory_manager.dram_data is None:
            raise ValueError("We Cannot dump dram for this communication type")
        # dump FMEM
        prefix = os.path.join(output_dir, 'sram')
        fmem_info_file = os.path.join(output_dir, 'fmem_info.txt')
        fmem_file = prefix + '_dump_f_{}.dat'.format(self.manager.frame_num)
        fmem_text_file = prefix + '_dump_f_{}.txt'.format(self.manager.frame_num)
        wmem_file = prefix + '_dump_w.dat'
        wmem_text_file = prefix + '_dump_w.txt'
        bmem_file = prefix + '_dump_b.dat'
        bmem_text_file = prefix + '_dump_b.txt'
        tmem_file = prefix + '_dump_t_{}.dat'.format(self.manager.frame_num)
        tmem_text_file = prefix + '_dump_t_{}.txt'.format(self.manager.frame_num)
        mm = self.memory_manager
        if not output_only:
            fmem_last_mapping = dict()
            fmem_info = self.layer_info.control_info.fmem_info
            for behavior in self.layer_info.control_info.behavior_info:
                if behavior.type == 'LOAD':
                    tensor = fmem_info.input_mapping[behavior.input_name]
                    idx, pivot_x, tail_x = tensor[behavior.index]
                    fmem_last_mapping[idx] = {'total_size': (tail_x - pivot_x) * tensor.yz_plane_size,
                                          'z_size': tensor.shape[-1], 'init_z_size': tensor.init_shape[-1]}
            for tensor in fmem_info.output_mapping.values():
                plane_size = tensor.yz_plane_size
                for mapping in tensor:
                    idx, pivot_x, tail_x = mapping
                    fmem_last_mapping[idx] = {'total_size': (tail_x - pivot_x) * plane_size, 'z_size': tensor.shape[-1],
                                          'init_z_size': tensor.init_shape[-1]}
            with open(fmem_info_file, 'w') as f:
                fmem_bank_size = self.manager.config.MIDAP.FMEM.NUM_ENTRIES
                fmem_last_mapping = dict(sorted(fmem_last_mapping.items()))
                f.write(str(len(fmem_last_mapping)) + '\n')
                for i in fmem_last_mapping:
                    f.write('{} {} {} {}\n'.format(i * fmem_bank_size, fmem_last_mapping[i]['total_size'],
                                                   fmem_last_mapping[i]['init_z_size'], fmem_last_mapping[i]['z_size']))
            mm.wmem.transpose(1, 0, 2).reshape(-1).tofile(wmem_file)
            mm.wmem.astype(np.uint8).transpose(1, 0, 2).reshape(-1).tofile(wmem_text_file, " ", "%02x")
            mm.bmem.reshape(-1).tofile(bmem_file)
            mm.bmem.astype(np.uint8).reshape(-1).tofile(bmem_text_file, " ", "%02x")
        mm.fmem.reshape(-1).tofile(fmem_file)
        mm.fmem.astype(np.uint8).reshape(-1).tofile(fmem_text_file, " ", "%02x")
        if self.manager.config.DRAM.COMM_TYPE != 'TEST_3D':
            mm.tmem.reshape(-1).tofile(tmem_file)
            mm.tmem.astype(np.uint8).reshape(-1).tofile(tmem_text_file, " ", "%02x")
        check_dram_data_info = []
        om = self.layer_info.control_info.fmem_info.output_mapping
        for name, mapping in om.items():
            if name not in mm.dram_dict:
                continue
            dt, addr = mm.dram_dict[name]
            if dt in [MemoryType.Input.value, MemoryType.Constant.value]:
                raise ValueError("Invalid Output data.. output data must be mapped to feature map address space: " + name)
            size = np.prod(mapping.shape[-2:]) * (mapping.shape[-3] - mapping.write_on_dram_pivot)
            addr += np.prod(mapping.shape[-2:]) * mapping.write_on_dram_pivot
            output_vtensor = self.layer_info.modules[0].output[0]
            addr += output_vtensor.get_address(output_vtensor.offset)
            check_dram_data_info.append((dt, addr, size, output_vtensor.init_shape[-1], output_vtensor.orig_shape[-1],
                                         mm.dram_data[dt][addr:addr+size]))
        if not output_only:
            info_file = os.path.join(output_dir, 'dram_info.txt')
            with open(info_file, 'w') as f:
                f.write("{}\n".format(len(check_dram_data_info)))
                for dt, addr, size, channel_size, channel_offset, _ in check_dram_data_info:
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(dt, addr, size, channel_size, channel_offset))
        if(len(check_dram_data_info) > 0):
            dram_data_file = os.path.join(output_dir, 'dram_check_{}.bin'.format(self.manager.frame_num))
            dram_text_file = os.path.join(output_dir, 'dram_check_{}.txt'.format(self.manager.frame_num))

            data_to_save = np.concatenate([x[-1] for x in check_dram_data_info])
            data_to_save.tofile(dram_data_file)
            data_to_save.astype(np.uint8).tofile(dram_text_file, " ", "%02x")
