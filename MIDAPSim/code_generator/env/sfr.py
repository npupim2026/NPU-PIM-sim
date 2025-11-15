from data_structure.attrdict import AttrDict
import numpy as np

# Special Function Registers, 32bits per each

__R = AttrDict()
reg = __R

## Register 0 ~ 3: reserved for triggering & feedback
# Trigger csg (control_signal_generator)
__R.tc = AttrDict()
__R.tc.no = 0
__R.tc.bits = [16, 16]
__R.tc.description = "Trigger Control Signal Generator) {} for trigger logic type and  trigger id".format(__R.tc.bits)
__R.trigger_csg = __R.tc

# Trigger dma #
__R.td = AttrDict()
__R.td.no = 1
__R.td.bits = [8, 24]
__R.td.description = "Trigger DMA) {} for trigger type and trigger id".format(__R.td.bits)
__R.trigger_dma = __R.td

# Trigger WDMA
__R.tw = AttrDict()
__R.tw.no = 3
__R.tw.bits = [8, 24]
__R.tw.description = "Trigger WDMA) {} for trigger type and trigger id".format(__R.td.bits)
__R.trigger_wdma = __R.tw

# feedback from control signal generator
__R.ffc = 48
__R.feedback_from_csg = 48

# feedback from dma controller
__R.ffd = 49
__R.feedback_from_dma = 49

# sram busy table: feedback from csg ## NOT IN USE... for debug
__R.f_sbt = 50
__R.feedback_sram_busy_table = 50

# Communicate with MCU to control WDMA
__R.ffw = 51
__R.feedback_from_wdma = 51

# Layer Parameter 
__R.lps = AttrDict()
__R.lps.no = 4
__R.lps.bits = [3, 2, 1, 5, 1, 5, 5, 1]
__R.lps.description = "layer_parameters) {} for cim_type, act_type, bias_type, bias_shift, quant_type, quant_shift1, quant_shift2, reduction_type".format(__R.lps.bits)
__R.layer_parameters = __R.lps

## Register 5 ~ 8: reserved for fixed offsets per layer

# fyo, wyo
__R.ro1 = AttrDict()
__R.ro1.no = 5
__R.ro1.bits = [16, 16]
__R.ro1.description = "read_offset1) {} for fyo, wyo".format(__R.ro1.bits)
__R.read_offset1 = __R.ro1

# fzo, wzo
__R.ro2 = AttrDict()
__R.ro2.no = 6
__R.ro2.bits = [16, 16]
__R.ro2.description = "read_offset2) {} for fzo, wzo".format(__R.ro2.bits)
__R.read_offset2 =__R.ro2

# fio, wio
__R.ro3 = AttrDict()
__R.ro3.no = 7
__R.ro3.bits = [16, 16]
__R.ro3.description = "read_offset3) {} for fio, wjo".format(__R.ro3.bits)
__R.read_offset3 =__R.ro3

# fjo (12), wjo (12), fko(4), wko(4)
__R.ro4 = AttrDict()
__R.ro4.no = 8
__R.ro4.bits = [12, 12, 4, 4]
__R.ro4.description = "read_offset4) {} for fjo, wjo, fko, wko".format(__R.ro4.bits)
__R.read_offset4 =__R.ro4

# read pivots # 16bits: fmem pivot, 16bits: wmem pivot
__R.rp = AttrDict()
__R.rp.no = 9
__R.rp.bits = [16, 16]
__R.rp.description = "read_pivot) {} for rfp, rwp".format(__R.rp.bits)
__R.read_pivot = __R.rp

# jump information (jump_valid, jump x, jump pivot)
__R.ji = AttrDict()
__R.ji.no = 10
__R.ji.bits = [8, 8, 16]
__R.ji.description = "jump_info) {} for jump_valid, jump_x, jump_pivot".format(__R.ji.bits)
__R.jump_info = __R.ji

# Static offset - differs per target kernel , 16bits: fso, 16bis: wso
__R.rso = AttrDict()
__R.rso.no = 11
__R.rso.bits = [16, 16]
__R.rso.description = "static_offset) {} for fso, wso".format(__R.rso.bits)
__R.read_static_offset = __R.rso

# # No Reset read & write offset # Current, reset write offset only, Deprecated
# __R.nrof = 12
# __R.no_reset_offset_flag = 12

# out-loop-info (y_iter, in_z_iter (output filter tile))
__R.oli = AttrDict()
__R.oli.no = 13
__R.oli.bits = [16, 16]
__R.oli.description = "outer_loop_info) {} for y-axis tier, z-axis iter for output".format(__R.oli.bits)
__R.outer_loop_info = __R.oli

# inter-loop info (kx, ky, kz)
__R.ili = AttrDict()
__R.ili.no = 14
__R.ili.bits = [8, 8, 16]
__R.ili.description = "inter_loop_info) {} for kx, ky, kz".format(__R.ili.bits)
__R.inter_loop_info = __R.ili

# Conv-YZ information wo, fo, jfo, fjo, yo, yzo
__R.cyzi = AttrDict()
__R.cyzi.no = 15
__R.cyzi.bits = [4, 4, 4, 4, 4, 4, 4, 4]
__R.cyzi.description = "conv_yz_offset_info) {} for wo, fo, jfo, fjo, yo, yzo, del_f, del_b".format(__R.cyzi.bits)
__R.conv_yz_information = __R.cyzi

# Tensor Virtualization Information) use_tv, scale, xoffset, yoffset, xstride, ystride
#__R.tvi = AttrDict()
#__R.tvi.no = 16
#__R.tvi.bits = [4, 4, 4, 4, 4, 4, 4, 4]
#__R.tvi.description = "Tensor Virtualization Info) {} for use_tv, scalex, scaley, xpivot, ypivot, ystride_offset, xoffset, yoffset"

# __R.tvi = AttrDict()
# __R.tvi.no = 16
# __R.tvi.bits = [4, 4, 4, 4, 4, 4] 
# __R.tvi.description = "Tensor Virtualization Info) {} for use_tv, scale, xpivot, ypivot, ystride_offset, xoffset, yoffset"

__R.tvi1 = AttrDict()
__R.tvi1.no = 16
__R.tvi1.bits = [8, 8, 8, 8]
__R.tvi1.description = "Tensor Virtualization Info) {} for use_tv, scalex, scaley, xpivot"

__R.tvi2 = AttrDict()
__R.tvi2.no = 17
__R.tvi2.bits = [8, 8, 8, 8]
__R.tvi2.description = "Tensor Virtualization Info) {} for ypivot, ystride_offset, xoffset, yoffset"

# write flag # 2bits / tmem_pivot_addr 14bits / fmem_pivot_addr 16bits
__R.wi = AttrDict()
__R.wi.no = 18
__R.wi.bits = [2, 14, 16]
__R.wi.description = "write_info) {} for wflag, wtp, wfp".format(__R.wi.bits)
__R.write_info = __R.wi

# write offsets: fmem write y-axis offset w/ 16bits, tmem/fmem write offset z-axis 16bits
__R.wo1 = AttrDict()
__R.wo1.no = 19
__R.wo1.bits = [16, 16]
__R.wo1.description = "write_offset1) {} for wfyo, wzo".format(__R.wo1.bits)
__R.write_offset1 = __R.wo1

# tmem write offset, y-axis 16bits: wtyo
__R.wo2 = 20
__R.write_offset2 = __R.wo2

# Bias info # Read addr, offset
__R.bi = AttrDict()
__R.bi.no = 21
__R.bi.bits = [16, 16]
__R.bi.description = "bias_info) {} for rbp, bzo".format(__R.bi.bits)
__R.bias_info = __R.bi

# replication padding
__R.rpad = AttrDict()
__R.rpad.no = 22
__R.rpad.bits = [4, 7, 7, 7, 7]
__R.rpad.description = "replication padding) {} for use_rpad, rpad_l, rpad_r, rpad_t, rpad_b"
__R.replication_padding = __R.rpad

## DMA Triggering Informations

# MIDAP SRAM "Busy" table (fmem * 4, wmem * 2, bmem * 2, tmem * 2) - 10 bits are required
__R.sbt = 23
__R.sram_busy_table = 23
### not in use in generated opcodes ... 

__R.DMA = AttrDict()

# request memory id
__R.DMA.mid = 24
__R.DMA.memory_id = 24

# Dest Pivot Address
__R.DMA.dpa = 25
__R.DMA.dst_pivot_address = 25

# Src Pivot Address
__R.DMA.spa = 26
__R.DMA.src_pivot_address = 26

# Transfer Unit Size
__R.DMA.tus = 27
__R.DMA.transfer_unit_size = 27

# # of Transfers
__R.DMA.nt = 28
__R.DMA.num_transfers = 28

# Src Transfer Offset
__R.DMA.sto = 29
__R.DMA.src_transfer_offset = 29

# Dst Transfer Offset
__R.DMA.dto = 30
__R.DMA.dst_transfer_offset = 30

class SFRSpace():
    def __init__(self, num_sfrs = 80, dtype = np.uint32):
        self.sfrs = np.array([-1 for _ in range(num_sfrs)]).astype(dtype)
    
    def intialize(self):
        for i in range(self.sfr.size):
            self.sfr[i] = -1
    
    def get_value(self, reg_id):
        if not isinstance(reg_id, int):
            return self.get_value_bits(reg_id.no, reg_id.bits)
        return self.sfrs[reg_id]

    def set(self, reg_id, value):
        self.sfrs[reg_id] = value

    def get_value_bits(self, reg_no, reg_bits):
        value = self.sfrs[reg_no]
        ret = []
        for bit in reg_bits:
            ret.append(value & (2**bit - 1))
            value = value >> bit
        return ret
    
    @property
    def layer_parameters(self):
        return self.get_value(reg.lps)

    @property
    def cim_type(self):
        return self.layer_parameters[0]
    
    @property
    def act_type(self):
        return self.layer_parameters[1]
    
    @property
    def bias_type(self):
        return self.layer_parameters[2]
    
    @property
    def bias_shift(self):
        return self.layer_parameters[3]
    
    @property
    def quant_type(self):
        return self.layer_parameters[4]
    
    @property
    def main_shift(self):
        return self.layer_parameters[5]
    
    @property
    def act_shift(self):
        return self.layer_parameters[6]
    
    @property
    def reduction_type(self):
        return self.layer_parameters[7]

