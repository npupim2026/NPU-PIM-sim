from __future__ import absolute_import, division, print_function, unicode_literals
from data_structure.instruction_components import SLayerInfo
from software.compiler.wmem_info import ComputeType
from software.system_compiler.memory_info import MemoryType

import numpy as np
import logging
import copy
import math

from data_structure.attrdict import AttrDict
from config import cfg

from code_generator.env import rule
from code_generator.env.rule import TriggerType
from code_generator.env.opcode import set_reg, reset_reg, wait_reg_eq
from code_generator.env.sfr import reg


class TransferInfo():
    def __init__(
        self,
        transfer_type,
        mem_id,
        src_addr,
        dst_addr,
        transfer_unit_size,
        num_transfers,
        src_transfer_offset = 0,
        dst_transfer_offset = 0
        ):
        self.transfer_type = transfer_type
        self.mem_id = mem_id
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.transfer_unit_size = transfer_unit_size
        self.num_transfers = num_transfers
        self.src_transfer_offset = src_transfer_offset
        self.dst_transfer_offset = dst_transfer_offset
    
    def to_nparray(self):
        return np.array(
            [self.transfer_type,
            self.mem_id,
            self.src_addr,
            self.dst_addr,
            self.transfer_unit_size,
            self.num_transfers,
            self.src_transfer_offset,
            self.dst_transfer_offset],
            dtype = np.int32)

class MemoryOpModule():
    def __init__(self, generator):
        # Set FMEM constraints
        self.generator = generator
        config = generator.config
        self.dynamic_base_addr = generator.dynamic_base_addr
        self.memory_type_to_transfer_type = {
            MemoryType.Shared.value: rule.TransferType.Shared,
            MemoryType.Constant.value: rule.TransferType.Constant if self.dynamic_base_addr else rule.TransferType.Normal,
            MemoryType.Temporal.value: rule.TransferType.Temporal if self.dynamic_base_addr else rule.TransferType.Normal,
            MemoryType.Input.value: rule.TransferType.IORead,
            MemoryType.Output.value: rule.TransferType.IOWrite,
        }
        self.config = config
        self.num_fmem = config.MIDAP.FMEM.NUM
        self.fmem_size = config.MIDAP.FMEM.NUM_ENTRIES
        # input = 0, output = 1, else = -1
        self.system_width = config.MIDAP.SYSTEM_WIDTH
        # Set WMEM constraints
        self.num_wmem = config.MIDAP.WMEM.NUM
        self.wmem_size = config.MIDAP.WMEM.NUM_ENTRIES
        # double buffered WMEM
        self.wmem_in_use = -1
        # BMEM setting
        self.bmem_size = config.MIDAP.BMEM.NUM_ENTRIES
        # double buffered BMEM
        self.bmem_in_use = -1
        self.tmem_size = config.MIDAP.WRITE_BUFFER.NUM_ENTRIES
        self.tmem_in_use = 0
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('gen')
        self.dram_dict = None
        self.dram_offset = [self.config.DRAM.OFFSET.SHARED, self.config.DRAM.OFFSET.INPUT, self.config.DRAM.OFFSET.OUTPUT, self.config.DRAM.OFFSET.WEIGHT_BIAS, self.config.DRAM.OFFSET.BUFFER]
        self.sram_offset_info = rule.get_sram_offset_info(config)
        self.request_id = -1
        self.request_array = [TransferInfo(0, 0, 0, 0, 0, 0)]

    @property
    def tid_offset(self):
        return self.sram_offset_info.tmem_id_offset

    @property
    def wid_offset(self):
        return self.sram_offset_info.wmem_id_offset
    
    @property
    def bid_offset(self):
        return self.sram_offset_info.bmem_id_offset

    @property
    def fid_offset(self):
        return self.sram_offset_info.fmem_id_offset
    
    @property
    def lid_offset(self):
        return self.sram_offset_info.lut_id_offset

    @property
    def svar(self):
        return self.generator.svar
    
    @property
    def soi(self):
        return self.sram_offset_info

    def get_request_array(self):
        array = np.zeros(0, dtype = np.int32)
        for item in self.request_array:
            array = np.concatenate([array, item.to_nparray()], axis = 0)
        return array

    def set_dram_info(self, dram_dict, *args, **kwargs):
        self.dram_dict = dram_dict
    
    def get_dram_address(self, name, offset):
        td, addr = self.dram_dict[name]
        if self.dynamic_base_addr:
            return addr + offset
        else:
            return (self.dram_offset[td] + addr + offset)
    
    def reset_wmem(self):
        self.logger.debug("Reset_WMEM")
        self.wmem_in_use = -1

    def switch_wmem(self):
        self.wmem_in_use = (self.wmem_in_use + 1) % 2
        self.logger.debug("Switch_WMEM: to "+str(self.wmem_in_use))
    
    def switch_bmem(self):
        self.bmem_in_use = (self.bmem_in_use + 1) % 2
        self.logger.debug("Switch_BMEM: to "+str(self.bmem_in_use))
    
    def switch_tmem(self):
        self.tmem_in_use = (self.tmem_in_use + 1) % 2
        self.logger.debug("Switch_TMEM: to "+str(self.tmem_in_use))
    
    def gen_dma_transfer_code(
            self,
            mem_id,
            transfer_unit_size,
            num_transfers,
            dram_pivot_address,
            sram_pivot_address,
            dram_transfer_offset,
            sram_transfer_offset,
            transfer_type = 0,
            write = False,
            ):
        transfer_code = []
        if(not write):
            transfer_info = TransferInfo(
                transfer_type,
                mem_id,
                dram_pivot_address,
                sram_pivot_address,
                transfer_unit_size,
                num_transfers,
                dram_transfer_offset,
                sram_transfer_offset)
        else:
            transfer_info = TransferInfo(
                transfer_type,
                mem_id,
                sram_pivot_address,
                dram_pivot_address,
                transfer_unit_size,
                num_transfers,
                sram_transfer_offset,
                dram_transfer_offset
                )
        request_id = len(self.request_array)
        self.request_array.append(transfer_info)
        # transfer_code += bitset(reg.sbt, mem_id)
        # transfer_code += bitwait(reg.f_sbt, mem_id) It must be driven by DMA Controller!!
        transfer_src = TriggerType.DMA_RUN
        transfer_code += set_reg(reg.trigger_dma.no, (request_id << reg.trigger_dma.bits[0]) + transfer_src)
        transfer_code += wait_reg_eq(reg.ffd, request_id)
        return transfer_code

    def gen_load_wmem_code(self, to_all, filter_name, filter_size, wmem_offset = 0, dram_offset = 0):
        if not (filter_size % 64 == 0 and wmem_offset % 64 == 0):
            raise ValueError("filter {}, size {}, offset {}".format(filter_name, filter_size, wmem_offset))
        wmem_idx = (self.wmem_in_use+1) % 2
        mem_id = self.wid_offset + wmem_idx
        transfer_unit_size = filter_size
        num_transfers = 1 if not to_all else self.num_wmem
        dram_pivot_address = self.get_dram_address(filter_name, dram_offset)
        transfer_type = self.memory_type_to_transfer_type[self.dram_dict[filter_name][0]]
        dram_transfer_offset = filter_size
        sram_transfer_offset = self.soi.wmem_addr_offset * 2
        sram_pivot_address = self.soi.wmem_pivot_address + self.soi.wmem_addr_offset * wmem_idx + wmem_offset
        return self.gen_dma_transfer_code(
                mem_id,
                transfer_unit_size,
                num_transfers,
                dram_pivot_address,
                sram_pivot_address,
                dram_transfer_offset,
                sram_transfer_offset,
                transfer_type,
                False
                )

    def gen_load_fmem_code(self, fmem_idx, data_name, data_size, fmem_offset = 0, dram_offset = 0):
        assert data_size % self.num_wmem == 0 and fmem_offset == 0 # fmem_offset > 0 is not supported yet
        mem_id = self.fid_offset + fmem_idx
        dram_address = self.get_dram_address(data_name, dram_offset)
        transfer_type = self.memory_type_to_transfer_type[self.dram_dict[data_name][0]]
        sram_address = self.soi.fmem_pivot_address + self.soi.fmem_addr_offset * fmem_idx
        return self.gen_dma_transfer_code(
                mem_id,
                data_size,
                1,
                dram_address,
                sram_address,
                0,
                0,
                transfer_type,
                False
                )

    def gen_load_bmem_code(self, bias_name, bias_size):
        bmem_idx = (self.bmem_in_use + 1) % 2
        mem_id = self.bid_offset + bmem_idx
        dram_address = self.get_dram_address(bias_name, 0)
        transfer_type = self.memory_type_to_transfer_type[self.dram_dict[bias_name][0]]
        sram_address = self.soi.bmem_pivot_address + self.soi.bmem_addr_offset * bmem_idx
        return self.gen_dma_transfer_code(
                mem_id,
                bias_size,
                1,
                dram_address,
                sram_address,
                0,
                0,
                transfer_type,
                False
                )

    def gen_load_lut_code(self, lut_name):
        lut_size = self.config.MIDAP.LUT.NUM_ENTRIES * 4
        mem_id = self.lid_offset
        dram_address = self.get_dram_address(lut_name, 0)
        transfer_type = self.memory_type_to_transfer_type[self.dram_dict[lut_name][0]]
        sram_address = self.soi.lut_pivot_address
        return self.gen_dma_transfer_code(
                mem_id,
                lut_size,
                1,
                dram_address,
                sram_address,
                0,
                0,
                transfer_type,
                False
                )

    def tmem_to_dram(
        self,
        data_name,
        dram_pivot_address,
        transfer_unit_size,
        transfer_offset,
        num_transfers_per_plane,
        tmem_pivot_address = 0,
        *args,
        **kwargs
        ):
        tmem_idx = self.tmem_in_use % 2
        mem_id = self.tid_offset + tmem_idx
        dram_pivot_address = self.get_dram_address(data_name, dram_pivot_address)
        dram_transfer_offset = transfer_offset
        sram_pivot_address = self.soi.tmem_pivot_address + self.soi.tmem_addr_offset * tmem_idx + tmem_pivot_address
        sram_transfer_offset = transfer_unit_size
        transfer_type = self.memory_type_to_transfer_type[self.dram_dict[data_name][0]]
        return self.gen_dma_transfer_code(
                mem_id,
                transfer_unit_size,
                num_transfers_per_plane,
                dram_pivot_address,
                sram_pivot_address,
                dram_transfer_offset,
                sram_transfer_offset,
                transfer_type,
                True
                )

    def get_wmem_addr(self, address):
        return ((self.wmem_in_use % 2) << (self.svar.rw_addr_bits + self.svar.read_offset_bits)) + (address >> self.svar.read_address_bits)
        
    def get_fmem_addr(self, bank_idx, address, write = False):
        if write:
            return (bank_idx << self.svar.wf_addr_bits) + (address >> self.svar.write_bits)
        else:
            return (bank_idx << (self.svar.rf_addr_bits + self.svar.read_offset_bits)) + (address >> self.svar.read_address_bits)

    def get_bmem_addr(self, address):
        return ((self.bmem_in_use % 2) << self.svar.channel_bits) + (address >> self.svar.write_bits)

    def get_tmem_addr(self, address):
        return ((self.tmem_in_use % 2) << self.svar.wt_addr_bits) + (address >> self.svar.write_bits)

class MemoryOpGenerator():
    def __init__(self, generator):
        config = generator.config
        self.generator = generator
        self.config = config
        self.system_width = config.MIDAP.SYSTEM_WIDTH
        self.filter_name = None
        self.bias_name = None
        self.compute_type = None
        self.input_tensor = None
        self.num_filters = 0
        self.load_filter_once = False
        self.all_filters_on_wmem = False
        self.filter_size = 0
        self.prepare_info = None
        self.num_wmem = config.MIDAP.WMEM.NUM
        self.filter_group_size = 1
        self.channel_broadcast = False
        self.next_filters_on_wmem = False
        self.memory_op_module = MemoryOpModule(generator) if self.config.DRAM.COMM_TYPE != 'TEST_3D' else Memory3DOpModule(generator)
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('gen')

    def set_dram_info(self, dram_address_dict):
        return self.memory_op_module.set_dram_info(dram_address_dict)
    
    def gen_init_code(self):
        init_code = []
        init_code += reset_reg(reg.trigger_dma)
        init_code += reset_reg(reg.trigger_wdma)
        # init_code += reset_reg(reg.sram_busy_table)
        return init_code

    def setup(self, layer_info : SLayerInfo):
        control_info = layer_info.control_info
        if isinstance(self.memory_op_module, Memory3DOpModule):
            self.memory_op_module.layer_index += 1
        # Load fmem information
        self.layer_info = layer_info
        self.input_mapping = control_info.fmem_info.input_mapping
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
        if self.compute_type == ComputeType.StdConv: # Conv
            self.filter_set_size = self.num_wmem
        elif self.compute_type in [ComputeType.DWConv, ComputeType.Pool, ComputeType.Elwise, ComputeType.WeightedSum]: # DWConv, Pool
            self.filter_set_size = 1
            self.use_extended_cim = True
        # elif self.compute_type == ComputeType.Elwise: # ArithmeticOp
        #     self.filter_set_size = 1
        #     self.use_extended_cim = True
        self.num_filters = self.num_filters // self.filter_set_size
        # Group related terms
        self.filter_group_size = wi.filter_group_size
        self.num_filter_groups = math.ceil(self.num_filters/self.filter_group_size)
        self.filter_offset = 0
        self.group_idx = -1
        # Filter loading configuration
        self.reverse_load = wi.reverse_load
        self.load_filter_once = wi.load_filter_once
        self.all_filters_on_wmem = self.filter_name is None
        self.channel_broadcast = wi.compute_type == ComputeType.WeightedSum # TODO: There may be more operations require this option
        # preparation information
        setup_code = []
        if self.filter_name is not None:
            if not self.wmem_info.prepared:
                self.memory_op_module.reset_wmem()
                setup_code += self.load_wmem(False)
                if self.bias_name is not None:
                    setup_code += self.memory_op_module.gen_load_bmem_code(self.bias_name, wi.bias_size)
            elif self.next_filters_on_wmem:
                self.all_filters_on_wmem = True
                self.next_filters_on_wmem = False
        if wi.lut_name is not None:
            setup_code += self.memory_op_module.gen_load_lut_code(wi.lut_name)
        self.prepare_info = wi.prepare_info
        if self.prepare_info is not None and self.load_filter_once and self.prepare_info.filter_name == self.filter_name:
            self.next_filters_on_wmem = True
        setup_code += self.setup_bmem()
        self.logger.debug(f"{self}")
        return setup_code

    def gen_fin_code(self):
        if isinstance(self.memory_op_module, Memory3DOpModule):
            self.memory_op_module.set_last_flag()
        return []
    
    def __repr__(self):
        ret = ""
        ret += "=========================================\n"
        ret += f"MemoryOpGenerator initialization information for Layer {self.layer_info.name}\n"
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
    # WMEM related functions
    def set_next(self, last_use = False, z_iters = 1):
        self.logger.debug("====================================================")
        self.logger.debug(f"set_next({(last_use, z_iters)}) is called")
        self.logger.debug(f"Current Status: [foffset = {self.filter_offset}, group_idx: {self.group_idx}]")
        if isinstance(self.memory_op_module, Memory3DOpModule):
            self.memory_op_module.csg_progress += 1
        next_foffset = self.filter_offset + z_iters
        check = next_foffset + self.group_idx * self.filter_group_size >= self.num_filters
        load_next = self.group_idx < 0
        code = []
        if self.compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum]:
            if next_foffset >= self.filter_group_size or check:
                load_next = True
        elif self.compute_type == ComputeType.Pool:
            if next_foffset >= self.num_filters:
                next_foffset = 0
        else:
            load_next = True
        if load_next:
            code, _, _ = self._set_next(last_use)
            self.filter_offset = 0
        else:
            self.filter_offset = next_foffset
        curr_idx = self.group_idx * self.filter_group_size + self.filter_offset
        last = curr_idx + z_iters >= self.num_filters
        self.logger.debug(f"Internal Info: [load_next = {load_next}]")
        self.logger.debug(f"Updated Status: [foffset = {self.filter_offset}, group_idx: {self.group_idx}]")
        self.logger.debug(f"Return: [code = {code}, last = {check}, curr_idx: {curr_idx}]")
        self.logger.debug("====================================================")
        return code, last, curr_idx    

    def _set_next(self, last_use=False):
        self.group_idx = (self.group_idx + 1) % self.num_filter_groups
        last_group = self.group_idx + 1 == self.num_filter_groups
        load_code = []
        if self.compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum]:
            switch_wmem = True
            if self.all_filters_on_wmem and self.group_idx == 0:
                switch_wmem = self.num_filter_groups % 2 == 0
            if switch_wmem:
                self.memory_op_module.switch_wmem()
            load_code = self.load_wmem(last_use)
        elif self.compute_type == ComputeType.Elwise:
            if not self.all_filters_on_wmem or self.num_filter_groups > 1:
                self.memory_op_module.switch_wmem()
            load_code = self.load_wmem(last_use)
        self.logger.debug("WMEM IN USE: {}".format(self.memory_op_module.wmem_in_use))
        filter_idx = self.group_idx * self.filter_group_size
        return load_code, last_group, filter_idx

    def load_wmem(self, last_use = False):
        load_filter_idx = (self.group_idx + 1) * self.filter_group_size
        load_prepare = False
        if load_filter_idx >= self.num_filters:
            if self.load_filter_once:
                self.all_filters_on_wmem = True
            load_prepare = last_use
            load_filter_idx = 0
        self.logger.debug("Load_WMEM: Load filter idx : {} to wmem {}, load_prepare : {}, all_filters_on_wmem: {}".format(load_filter_idx, (self.memory_op_module.wmem_in_use + 1) % 2,  load_prepare, self.all_filters_on_wmem))
        if not load_prepare:
            if self.all_filters_on_wmem:
                return []
            next_group_size = min(self.num_filters - load_filter_idx, self.filter_group_size) if not self.channel_broadcast else 1
            if self.compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum]:
                wmem_pivot = 0 if not self.load_filter_once else self.filter_group_size * (load_filter_idx // (2 * self.filter_group_size))
                load_code = self.load_filter(
                        self.compute_type,
                        self.filter_name,
                        self.filter_size,
                        next_group_size,
                        wmem_pivot,
                        load_filter_idx
                        )
            elif self.compute_type == ComputeType.Elwise:
                filter_idx_pivot = self.num_filters - load_filter_idx - next_group_size if self.reverse_load else load_filter_idx
                load_code = self.load_filter(
                        self.compute_type,
                        self.filter_name,
                        self.filter_size,
                        next_group_size,
                        0,
                        filter_idx_pivot,
                        self.reverse_load
                        )
        elif self.prepare_info is None or self.next_filters_on_wmem:
            return []
        else:
            self.logger.debug("Load Prepare Info")
            pi = self.prepare_info
            if pi.compute_type == ComputeType.Elwise:
                load_idx = 0 if not pi.reverse_load else pi.num_filters - pi.filter_group_size
                load_code = self.load_filter(
                        pi.compute_type,
                        pi.filter_name,
                        pi.filter_size,
                        pi.filter_group_size if pi.compute_type != ComputeType.WeightedSum else 1, # TODO: Temporal solution
                        0,
                        load_idx,
                        pi.reverse_load
                        )
            else:
                load_code = self.load_filter(
                        pi.compute_type,
                        pi.filter_name,
                        pi.filter_size,
                        pi.filter_group_size if pi.compute_type != ComputeType.WeightedSum else 1, # TODO: Temporal solution
                        0,
                        0,
                        )
        return load_code

    def load_filter(
            self,
            compute_type,
            filter_name,
            filter_size,
            group_size,
            wmem_pivot,
            filter_idx_pivot,
            reverse_load = False
            ):
        self.logger.debug("Function call: Load Filter")
        load_code = []
        if compute_type in [ComputeType.StdConv, ComputeType.DWConv, ComputeType.WeightedSum]:
            to_all = compute_type == ComputeType.StdConv
            load_code += self.memory_op_module.gen_load_wmem_code(
                    to_all = to_all,
                    filter_name = filter_name,
                    filter_size = filter_size * group_size,
                    wmem_offset = wmem_pivot * filter_size,
                    dram_offset = filter_idx_pivot * filter_size * self.filter_set_size,
                    )
        else:
            for g in range(group_size):
                if reverse_load:
                    pivot = filter_idx_pivot + group_size - g - 1
                else:
                    pivot = filter_idx_pivot + g
                load_code += self.memory_op_module.gen_load_wmem_code(
                        to_all = False,
                        filter_name = filter_name,
                        filter_size = filter_size,
                        wmem_offset = g * filter_size,
                        dram_offset = pivot * filter_size
                        )
        return load_code

    def get_wmem_addr(self, address):
        pivot = 0
        if self.load_filter_once:
            pivot = (self.group_idx //  2) * self.filter_group_size
        if self.compute_type != ComputeType.WeightedSum:
            pivot += self.filter_offset
        address = address + pivot * self.filter_size
        return self.memory_op_module.get_wmem_addr(address)

    def load_fmem(self, fmem_idx, data_name, info):
        inp = self.input_mapping[data_name]
        data_size = (info[1] - info[0]) * inp.yz_plane_size
        data_address = inp.yz_plane_size * info[0]
        load_code =  self.memory_op_module.gen_load_fmem_code(
                fmem_idx,
                data_name,
                data_size,
                0,
                data_address,
                )
        return load_code

    def get_fmem_addr(self, bank_idx, address, write = False):
        return self.memory_op_module.get_fmem_addr(bank_idx, address, write)
    
    def sync(self): ## TODO
        return []

    #""" BMEM related """
    def setup_bmem(self):
        code = []
        if self.bias_name is not None:
            self.memory_op_module.switch_bmem()
        if self.prepare_info is not None and self.prepare_info.bias_name is not None:
            pi = self.prepare_info
            code += self.memory_op_module.gen_load_bmem_code(pi.bias_name, pi.bias_size)
        return code

    def get_bmem_addr(self, address):
        return self.memory_op_module.get_bmem_addr(address)
    
    def get_tmem_addr(self, address):
        return self.memory_op_module.get_tmem_addr(address)
    
    def transfer_tmem(
            self,
            data_name,
            dram_pivot_address,
            transfer_unit_size,
            transfer_offset,
            num_transfers,  # per plane
            tmem_pivot_address = 0,
            num_plane = 1,
            plane_offset = 0,
            flip = False,
            ):
        # TODO: Modify transfer_info class
        code = self.memory_op_module.tmem_to_dram(
            data_name,
            dram_pivot_address,
            transfer_unit_size,
            transfer_offset,
            num_transfers,
            tmem_pivot_address,
            num_plane,
            plane_offset,
            flip
            )
        self.memory_op_module.switch_tmem()
        return code

class DMA3DDescriptor():
    num_of_fields = 0

    '''
    dram_addr_pivot = {
        rule.TransferType.Normal: 0,
        rule.TransferType.IORead: 0,
        rule.TransferType.IOWrite: 0,
        rule.TransferType.Instruction: cfg.DRAM.OFFSET.INSTRUCTION
    }
    '''

    def __init__(
        self, acnt, boff, bcnt,
        coff: int = 0,
        ccnt: int = 1,
        csub: bool = False,
        bsub: bool = False,
        bcsync: bool = False,
        last: bool = False,
        baddr_ddr: int = 0,
        baddr_sram: int = 0,
        mem_id = 0,
        transfer_type = 0,
        layer_index = -1
    ):
        self.acnt = int(acnt)
        self.boff = int(boff)
        self.bcnt = int(bcnt)
        self.coff = int(coff)
        self.ccnt = int(ccnt) 

        self.csub = csub
        self.bsub = bsub
        self.bcsync = bcsync
        self.last = last
        self.baddr_ddr_offset = int(baddr_ddr)
        self.baddr_sram = int(baddr_sram)
        self.mem_id = mem_id
        self.transfer_type = transfer_type
        self.layer_index = layer_index

    @property
    def data_size(self):
        return self.acnt * self.bcnt * self.ccnt

    @property
    def baddr_ddr(self):
        #return DMA3DDescriptor.dram_addr_pivot[self.transfer_type] + self.baddr_ddr_offset
        return self.baddr_ddr_offset

    def to_nparray(self):
        pass

class RDMA3DDescriptor(DMA3DDescriptor):
    num_of_fields = 8

    def to_nparray(self):
        # NEXT_DSC and INTENB field should be set manually
        # Assumed 2D descriptor
        dram_acnt = self.acnt * self.bcnt
        return np.array([
            self.transfer_type,
            self.mem_id,
            dram_acnt & 0xFFFF,
            (self.csub << 5) | (self.bsub << 4) | (self.bcsync << 2), #| self.last,
            self.baddr_ddr,
            self.baddr_sram >> 4,                   # 128 bit align
            (self.boff << 12) | (self.acnt >> 4),   # 128 bit align
            dram_acnt >> 16,                        # expansion register
        ],
        dtype = np.int32)

class WDMA3DDescriptor(DMA3DDescriptor):
    num_of_fields = 7

    def to_nparray(self):
        # NEXT_DSC and INTENB field should be set manually
        return np.array([
            self.transfer_type,
            self.layer_index,
            self.acnt,
            (self.boff << 16) | self.bcnt,
            (self.coff << 16) | self.ccnt,
            (self.csub << 5) | (self.bsub << 4) | (self.bcsync << 2) | self.last,
            self.baddr_ddr,
        ],
        dtype = np.int32)

class Memory3DOpModule(MemoryOpModule):
    def __init__(self, generator):
        super().__init__(generator)
        self.tmem_size = self.config.MIDAP.WRITE_BUFFER.NUM_ENTRIES * 2
        self.tmem_in_use = None     # Use a FIFO buffer instead of explicit TMEM
        self.rdma_request_array = [RDMA3DDescriptor(0, 0, 0, 1)]
        self.wdma_request_array = [WDMA3DDescriptor(0, 0, 0, 1, layer_index=0)]
        self.request_array = (self.rdma_request_array, self.wdma_request_array)
        self.layer_index = -1
        self.last_csg_progress = -1
        self.csg_progress = -1

    def switch_tmem(self):
        pass    # No explicit TMEM

    def gen_3d_dma_transfer_code(
        self,
        descriptor: DMA3DDescriptor = None,
    ):
        transfer_code = []
        #request_id = len(self.request_array)
        #self.request_array.append(descriptor)
        transfer_src = TriggerType.DMA_RUN
        if isinstance(descriptor, RDMA3DDescriptor):
            if self.csg_progress != self.last_csg_progress:
                if self.rdma_request_array:
                    self.rdma_request_array[-1].last = True
                self.last_csg_progress = self.csg_progress
            request_id = len(self.rdma_request_array)
            self.rdma_request_array.append(descriptor)
            transfer_code += set_reg(reg.trigger_dma.no, (request_id << reg.trigger_dma.bits[0]) + transfer_src)
            transfer_code += wait_reg_eq(reg.ffd, request_id)
        elif isinstance(descriptor, WDMA3DDescriptor):
            request_id = len(self.wdma_request_array)
            self.wdma_request_array.append(descriptor)
            transfer_code += wait_reg_eq(reg.ffw, request_id)
            transfer_code += set_reg(reg.trigger_wdma.no, (request_id << reg.trigger_wdma.bits[0]) + transfer_src)
        return transfer_code

    def gen_load_wmem_code(self, to_all, filter_name, filter_size, wmem_offset = 0, dram_offset = 0):
        if not (filter_size % 64 == 0 and wmem_offset % 64 == 0):
            raise ValueError("filter {}, size {}, offset {}".format(filter_name, filter_size, wmem_offset))
        wmem_idx = (self.wmem_in_use+1) % 2
        mem_id = self.wid_offset + wmem_idx
        descriptor = RDMA3DDescriptor(
            acnt=filter_size,
            boff=self.soi.wmem_addr_offset * 2,
            bcnt=1 if not to_all else self.num_wmem,
            baddr_ddr=self.get_dram_address(filter_name, dram_offset),
            baddr_sram=self.soi.wmem_pivot_address + self.soi.wmem_addr_offset * wmem_idx + wmem_offset,
            mem_id=mem_id,
            transfer_type=self.memory_type_to_transfer_type[self.dram_dict[filter_name][0]],
        )
        return self.gen_3d_dma_transfer_code(descriptor)

    def gen_load_fmem_code(self, fmem_idx, data_name, data_size, fmem_offset = 0, dram_offset = 0):
        assert data_size % self.num_wmem == 0 and fmem_offset == 0 # fmem_offset > 0 is not supported yet
        mem_id = self.fid_offset + fmem_idx
        descriptor = RDMA3DDescriptor(
            acnt=data_size,
            boff=data_size,
            bcnt=1,
            baddr_ddr=self.get_dram_address(data_name, dram_offset),
            baddr_sram=self.soi.fmem_pivot_address + self.soi.fmem_addr_offset * fmem_idx,
            mem_id=mem_id,
            transfer_type=self.memory_type_to_transfer_type[self.dram_dict[data_name][0]],
        )
        return self.gen_3d_dma_transfer_code(descriptor)

    def gen_load_bmem_code(self, bias_name, bias_size):
        bmem_idx = (self.bmem_in_use + 1) % 2
        mem_id = self.bid_offset + bmem_idx
        descriptor = RDMA3DDescriptor(
            acnt=bias_size,
            boff=bias_size,
            bcnt=1,
            baddr_ddr=self.get_dram_address(bias_name, 0),
            baddr_sram=self.soi.bmem_pivot_address + self.soi.bmem_addr_offset * bmem_idx,
            mem_id=mem_id,
            transfer_type=self.memory_type_to_transfer_type[self.dram_dict[bias_name][0]],
        )
        return self.gen_3d_dma_transfer_code(descriptor)

    def gen_load_lut_code(self, lut_name):
        lut_size = self.config.MIDAP.LUT.NUM_ENTRIES * 4
        mem_id = self.lid_offset
        descriptor = RDMA3DDescriptor(
            acnt=lut_size,
            boff=lut_size,
            bcnt=1,
            baddr_ddr=self.get_dram_address(lut_name, 0),
            baddr_sram=self.soi.lut_pivot_address,
            mem_id=mem_id,
            transfer_type=self.memory_type_to_transfer_type[self.dram_dict[lut_name][0]],
        )
        return self.gen_3d_dma_transfer_code(descriptor)

    def tmem_to_dram(
        self,
        data_name,
        dram_pivot_address,
        transfer_unit_size,
        transfer_offset,
        num_transfers_per_plane,
        tmem_pivot_address = 0,
        num_plane = 1,
        plane_offset = 0,
        flip = False,
        *args,
        **kwargs
    ):
        descriptor = WDMA3DDescriptor(
            acnt=transfer_unit_size,
            boff=transfer_offset,
            bcnt=num_transfers_per_plane,
            coff=plane_offset,
            ccnt=num_plane,
            bcsync=True,
            csub=flip,
            baddr_ddr=self.get_dram_address(data_name, dram_pivot_address),
            transfer_type=self.memory_type_to_transfer_type[self.dram_dict[data_name][0]],
            layer_index=self.layer_index
        )
        return self.gen_3d_dma_transfer_code(descriptor)

    def get_tmem_addr(self, address):
        return (address >> self.svar.write_bits)

    def get_request_array(self):
        rdma_array = np.zeros(0, dtype = np.int32)
        wdma_array = np.zeros(0, dtype = np.int32)
        for item in self.rdma_request_array:
            rdma_array = np.concatenate([rdma_array, item.to_nparray()], axis = 0)
        for item in self.wdma_request_array:
            wdma_array = np.concatenate([wdma_array, item.to_nparray()], axis = 0)
        return rdma_array, wdma_array

    def set_last_flag(self):
        if self.wdma_request_array and self.wdma_request_array[-1].layer_index == self.layer_index:
            self.wdma_request_array[-1].last = True
        if self.rdma_request_array:
            self.rdma_request_array[-1].last = True
