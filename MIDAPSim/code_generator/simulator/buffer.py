import numpy as np
from config import cfg
import math

class InstBuf():
    bits = {
            'phase': 2,
            'faddr': 14,
            'waddr': 9,
            'channel_idx': 8,
            'delete': 4,
            'align': 2,
            'reset': 1,
            'ignore': 1,
            'last': 1,
            'wflag': 3,
            'wfaddr': 16,
            'wtaddr': 11,
            }
    def __init__(
            self,
            items = None,
            svar = None):
        if items is None:
            items = list(self.bits.keys())
        assert set(list(self.bits.keys())).issuperset(set(items))
        self.reg = [0 for _ in range(len(items))]
        self.items = items
        self.svar = svar
        self.debug = svar is not None
        if svar is not None:
            self.bits['faddr'] = self.svar.rf_addr_bits + self.svar.fmem_bank_bits
            self.bits['waddr'] = self.svar.rw_addr_bits + 1
            self.bits['channel_idx'] = self.svar.channel_bits + 1
            self.bits['wfaddr'] = self.svar.wf_addr_bits + self.svar.fmem_bank_bits
            self.bits['wtaddr'] = self.svar.wt_addr_bits + 1
        self.item_to_id = {x: i for i, x in enumerate(items)}

    def set(self, key, value):
        if self.debug:
            assert value >= 0
            if value >= math.pow(2, self.bits[key]):
                value = int(value) & int(math.pow(2, self.bits[key]) - 1)
        self.reg[self.item_to_id[key]] = value

    def __getattr__(self, key):
        return np.uint32(self.reg[self.item_to_id[key]])

    def __repr__(self):
        if not self.debug:
            return "InstBuf Status: {}".format(self.reg)
        else:
            def rep(name, value):
                value = np.uint32(value)
                if name == 'faddr':
                    bank_idx = value >> self.svar.rf_addr_bits
                    row = value & self.svar.rf_addr_mask
                    return (bank_idx, row)
                elif name == 'waddr':
                    wmem_idx = value >> self.svar.rw_addr_bits
                    row = value & self.svar.rw_addr_mask
                    return (wmem_idx, row)
                elif name == 'wflag':
                    return [(value & 0x2) > 0, (value & 0x1) > 0]
                elif name == 'wfaddr':
                    offset_bits = self.svar.read_bits - self.svar.write_bits
                    bank_idx = value >> self.svar.wf_addr_bits
                    row = (value & self.svar.wf_addr_mask) >> offset_bits 
                    offset = (value & offset_bits) << self.svar.write_bits
                    return (bank_idx, row, offset)
                elif name == 'wtaddr':
                    offset_bits = self.svar.read_bits - self.svar.write_bits
                    tmem_idx = value >> self.svar.wt_addr_bits
                    #row = (value & self.svar.wt_addr_mask) >> offset_bits
                    #offset = (value & offset_bits) << self.svar.write_bits
                    addr = (value & self.svar.wt_addr_mask) << self.svar.write_bits
                    return (tmem_idx, addr)
                else:
                    return value
            ret = ''
            ret += "Bits per each: {}\n".format(list(self.bits.values()))
            ret += "Required InstBuf Buffer size: {} \n".format(sum([self.bits[x] for x in self.items]))
            ret += ','.join(["{}: {}".format(x, rep(x, self.reg[i])) for (i, x) in enumerate(self.items)])
            return ret
    
    def set_from(self, ins):
        if isinstance(ins, InstBuf):
            assert set(ins.items).issuperset(set(self.items))
            for x in ins.items:
                if x in self.items:
                    self.set(x, ins.__getattr__(x))
        elif isinstance(ins, dict):
            assert set(list(ins.keys())).issuperset(set(self.items))
            for x in ins:
                if x in self.items:
                    self.set(x, ins[x])

class DataBuf():
    def __init__(self, config):
        system_width = config.MIDAP.SYSTEM_WIDTH
        num_cims = config.MIDAP.WMEM.NUM
        self.fbuf = np.zeros(system_width*2, dtype=np.int32) # Hardware: int8
        self.broadcast_fbuf = np.zeros(system_width, dtype=np.int32) # Hardware: int8
        self.wbuf = np.zeros([num_cims, system_width], dtype=np.int32) # Hardware: int8
        self.alu_buf = np.zeros([num_cims, system_width], dtype=np.int32)
        self.csatree_buf = np.zeros(num_cims, dtype=np.int32)
        self.cim_obuf = np.zeros(num_cims, dtype=np.int32) # Must be located in each cim
        self.ecim_obuf = np.zeros(system_width, dtype=np.int32) # Must be located in ecim
        self.cim_act_ibuf = np.zeros(num_cims, dtype = np.int32)
        self.cim_act_tbuf = np.zeros(num_cims, dtype = np.int32)
        self.e_act_ibuf = np.zeros(system_width, dtype = np.int32)
        self.e_act_tbuf = np.zeros(system_width, dtype = np.int32)
        self.bias_buf = np.zeros(system_width, dtype = np.int32)
        self.activation_tbuf = np.zeros(system_width, dtype = np.int32)
        self.activation_obuf = np.zeros(system_width, dtype = np.int8)
        self.reduction_buf = np.zeros(config.MIDAP.REDUCTION.NUM_ENTRIES, dtype = np.int32)
        self.reduction_obuf = np.zeros(num_cims, dtype = np.int32)
        self.reduction_dirty_buf = np.ones(config.MIDAP.REDUCTION.NUM_ENTRIES // num_cims, dtype = np.uint8)
        self.write_buf = np.zeros(system_width, dtype = np.int8)
