import numpy as np
import copy
import logging

from code_generator.env import rule
from software.system_compiler.memory_info import MemoryType

from config import cfg

class Memory():
    def __init__(self, config, midap_manager):
        self.config = config
        self.manager = midap_manager
        self.mm = midap_manager.memory_controller.memory_manager
        self.num_fmem = self.config.MIDAP.FMEM.NUM
        self.fmem_size = self.config.MIDAP.FMEM.NUM_ENTRIES
        self.system_width = self.config.MIDAP.SYSTEM_WIDTH
        data_type = np.int8
        # Set WMEM constraints
        self.num_cims = self.config.MIDAP.WMEM.NUM
        self.wmem_size = self.config.MIDAP.WMEM.NUM_ENTRIES
        self.ewmem_size = self.config.MIDAP.WMEM.E_NUM_ENTRIES
        # double buffered WMEM
        # BMEM setting
        self.bmem_size = self.config.MIDAP.BMEM.NUM_ENTRIES
        # double buffered BMEM
        fill = 0
        self.fmem = np.full([self.num_fmem, self.fmem_size], fill, dtype = data_type)
        self.wmem = np.full([2, self.num_cims, self.ewmem_size], fill, dtype = data_type)
        self.bmem = np.full([2, self.bmem_size], fill, dtype = data_type)
        self.tmem = np.full([2, self.config.MIDAP.WRITE_BUFFER.NUM_ENTRIES], fill, dtype = data_type)
        self.lut_raw  = np.full([self.config.MIDAP.LUT.NUM_ENTRIES * 2 * 2], fill, dtype = np.uint8)
        self.dram_offset = dict(sorted({
            self.config.DRAM.OFFSET.SHARED: MemoryType.Shared.value,
            self.config.DRAM.OFFSET.INPUT: MemoryType.Input.value,
            self.config.DRAM.OFFSET.OUTPUT: MemoryType.Output.value,
            self.config.DRAM.OFFSET.WEIGHT_BIAS: MemoryType.Constant.value,
            self.config.DRAM.OFFSET.BUFFER: MemoryType.Temporal.value,
        }.items()))
        #Warning: assumption, dram_offset[0] < dram_offset[1] 
        self.dram_data = []
        self.diff_arr = []
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('debug')
        self.svar = rule.make_system_variables(config)
    
    @property
    def lut(self):
        return self.lut_raw.reshape(self.config.MIDAP.LUT.NUM_ENTRIES, -1)

    ############################ For Functionality Checking ###############################
    def copy_memory(self, on_chip = True, off_chip = True):
        if on_chip:
            self.fmem[:,:] = self.mm.fmem[:,:]
            self.wmem[:, :, :] = self.mm.wmem[:, :, :]
            self.bmem[:,:] = self.mm.bmem[:, :]
            self.tmem[:,:] = self.mm.tmem[:, :]
        if off_chip:
            if self.dram_data is not None:
                del self.dram_data
            self.dram_data = [copy.copy(x) for x in self.mm.dram_data]
    
    def compare(self):
        # FMEM
        error = False
        sub = np.abs(self.fmem - self.mm.fmem)
        self.diff_arr = np.where(sub > 0, 1, 0)
        diff_cnt = np.sum(self.diff_arr)
        if diff_cnt > 0:
            self.logger.info("Compare: diff in FMEM: {}".format(diff_cnt))
            error = True
        # DRAM
        modules = self.manager.layer_info.modules
        target_specs = []
        for m in modules:
            for ot in m.output:
                target_specs.append((ot.name, np.prod(np.array(ot.orig_shape))))
        for name, size in target_specs:
            if name not in self.mm.dram_dict:
                continue
            tp, addr = self.mm.dram_dict[name]
            sub = np.abs(self.mm.dram_data[tp][addr:addr+size] - self.dram_data[tp][addr: addr+size])
            diff_arr = np.where(sub > 0, 1, 0)
            diff_cnt = np.sum(diff_arr)
            if diff_cnt > 0:
                self.logger.info("Compare: diff in DRAM - name: {}, diff_cnt: {}".format(name, diff_cnt))
                error = True
                self.dram_diff_arr = diff_arr
        return error
    
    ################################## Called by Datapath #########################################
    def load_fbuf(self, fbuf, faddr):
        faddr = np.uint32(faddr)
        bank_idx = np.right_shift(faddr, self.svar.rf_addr_bits)
        loc = (faddr & self.svar.rf_addr_mask) * self.system_width ## np.left_shift(@ , 6)
        fbuf[:] = self.fmem[bank_idx, loc:loc+self.system_width]

    def load_wbuf(self, wbuf, waddr, to_all):
        idx = self.num_cims if to_all == 1 else 1
        waddr = np.uint32(waddr)
        db_idx = np.right_shift(waddr, self.svar.rw_addr_bits)
        loc = (waddr & self.svar.rw_addr_mask) * self.system_width ## np.left_shift(@ , 6)
        wbuf[:idx,:] = self.wmem[db_idx, :idx, loc:loc+self.system_width]

    def load_bbuf(self, bbuf, channel_idx):
        channel_idx = np.uint32(channel_idx)
        bmem_idx = channel_idx >> self.svar.channel_bits
        bmem_addr = (channel_idx & self.svar.channel_mask)
        bmem_addr = bmem_addr << (self.svar.write_bits)
        bmem_addr = (bmem_addr // self.system_width) * self.system_width
        bbuf[:] = self.bmem[bmem_idx, bmem_addr: bmem_addr + self.system_width]

    def get_lut_items(self, x1):
        lut_raw = self.lut[x1, :].astype(np.int16)
        lut = np.zeros([lut_raw.shape[0], 2])
        for i in range(2):
            lut[:, i] = np.left_shift(lut_raw[:, 2*i+1], 8) + lut_raw[:,2*i]
        return lut.astype(np.int32)

    def write(self, wbuf, wflag, wfaddr, wtaddr, size): # Assume that write validation is implemented as well..
        if wflag & 0x2 > 0:
            wfaddr = np.uint32(wfaddr)
            bank_idx = np.right_shift(wfaddr, self.svar.wf_addr_bits)
            loc = (wfaddr & self.svar.wf_addr_mask) * self.svar.write_unit
            self.fmem[bank_idx, loc:loc+size] = wbuf[:size]
        if wflag & 0x1 > 0:
            wtaddr = np.uint32(wtaddr)
            tmem_idx = np.right_shift(wtaddr, self.svar.wt_addr_bits)
            loc = (wtaddr & self.svar.wt_addr_mask) * self.svar.write_unit
            self.tmem[tmem_idx, loc:loc+size] = wbuf[:size]
   
    ## DRAM Data Wrapper ##
    def get_dram_data(self, address, size):
        for offset in reversed(self.dram_offset):
            if address >= offset:
                addr = address - offset
                return self.dram_data[self.dram_offset[offset]][addr:addr+size]

   ############################ Called by DMA ################################
    def write_dram_with_offset(self, tmem_idx, tmem_address, dram_address, size):
        for offset in reversed(self.dram_offset):
            if dram_address >= offset:
                addr = dram_address - offset
                self.dram_data[self.dram_offset[offset]][addr:addr+size] = self.tmem[tmem_idx, tmem_address:tmem_address + size]
                break

    def load_fmem(self, bank_idx, dram_addr, size):
        size = size
        self.logger.debug("Load FMEM: from DRAM Addr = {} to fmem[{}][:{}]".format(dram_addr, bank_idx, size))
        self.fmem[bank_idx, :size] = self.get_dram_data(dram_addr, size)

    def load_wmem(self, wmem_idx, wmem_addr, dram_addr, size):
        size = size
        widx = wmem_idx % 2
        idx = wmem_idx // 2
        self.logger.debug("Load WMEM: from DRAM Addr = {} to wmem[{},{},{}:{}]".format(dram_addr,widx, idx, wmem_addr, wmem_addr + size))
        self.wmem[widx, idx, wmem_addr:wmem_addr+size] = self.get_dram_data(dram_addr, size)

    def load_bmem(self, bmem_idx, bmem_addr, dram_addr, size):
        if bmem_addr > 0:
            self.logger.warning("Maybe load_bmem includes invalid offset")
        size = size
        self.logger.debug("Load BMEM: from DRAM Addr = {} to bmem[{},{}:{}]".format(dram_addr, bmem_idx, bmem_addr, bmem_addr+ size))
        self.bmem[bmem_idx, bmem_addr:bmem_addr + size] = self.get_dram_data(dram_addr, size)

    def load_lut(self, lut_addr, dram_addr, size):
        if lut_addr > 0:
            self.logger.warning("Maybe load_lut includes invalid offset")
        size = size
        self.logger.debug("Load LUT: from DRAM Addr = {} to lut[{}:{}]".format(dram_addr, lut_addr, lut_addr+ size))
        self.lut_raw[lut_addr:lut_addr + size] = self.get_dram_data(dram_addr, size)


class Memory3DDMA(Memory):
    def __init__(self, config, midap_manager):
        super().__init__(config, midap_manager)
        self.tmem = np.reshape(self.tmem, -1)   # No external tmem; write buffer is in the DMA
        self.buffer_write_pivot = 0             # Pivot for the circular buffer
        self.buffer_read_pivot = 0              # Pivot for the circular buffer
        self.write_available_space = self.tmem.size

    def copy_memory(self, on_chip = True, off_chip = True):
        if on_chip:
            self.fmem[:,:] = self.mm.fmem[:,:]
            self.wmem[:, :, :] = self.mm.wmem[:, :, :]
            self.bmem[:,:] = self.mm.bmem[:, :]
            self.tmem[:] = self.mm.tmem[:]
            self.buffer_write_pivot = self.mm.buffer_write_pivot
            self.buffer_read_pivot = self.mm.buffer_read_pivot
            self.write_available_space = self.mm.write_available_space
        if off_chip:
            if self.dram_data is not None:
                del self.dram_data
            self.dram_data = [copy.copy(x) for x in self.mm.dram_data]

    def write(self, wbuf, wflag, wfaddr, wtaddr, size):
        if wflag & 0x2 > 0:
            wfaddr = np.uint32(wfaddr)
            bank_idx = np.right_shift(wfaddr, self.svar.wf_addr_bits)
            loc = (wfaddr & self.svar.wf_addr_mask) * self.svar.write_unit
            self.fmem[bank_idx, loc:loc+size] = wbuf[:size]
        if wflag & 0x1 > 0:
            # wtaddr is ignored
            loc = self.buffer_write_pivot
            self.buffer_write_pivot = (loc + size) % self.tmem.size
            if loc + size > self.tmem.size:
                pivot = self.tmem.size - loc
                self.tmem[loc:] = wbuf[:pivot]
                self.tmem[:size - pivot] = wbuf[pivot:]
            else:
                self.tmem[loc:loc+size] = wbuf[:size]

    def write_dram_with_offset(self, tmem_idx, tmem_address, dram_address, size):
        # tmem_idx and tmem_address are ignored
        for offset in reversed(self.dram_offset):
            if dram_address >= offset:
                addr = dram_address - offset
                tmem_address = self.buffer_read_pivot
                self.buffer_read_pivot = (tmem_address + size) % self.tmem.size
                self.dram_data[self.dram_offset[offset]][addr:addr+size] = np.take(self.tmem, range(tmem_address, tmem_address + size), mode='wrap')
                break
