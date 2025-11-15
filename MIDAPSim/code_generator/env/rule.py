import math

from midap_backend.wrapper.op_wrapper import ConvPoolWrapper, ConvWrapper, DWWrapper, PoolWrapper, DWConvWrapper, ArithmeticWrapper, MulWrapper, AddWrapper, AvgpoolWrapper, MaxpoolWrapper, SumWrapper
from data_structure.attrdict import AttrDict
from enum import Enum, auto

def make_system_variables(config):
    var = AttrDict()
    system_width = config.MIDAP.SYSTEM_WIDTH
    num_wmem = config.MIDAP.WMEM.NUM
    num_fmem = config.MIDAP.FMEM.NUM
    var.read_unit = system_width
    var.write_unit = num_wmem
    var.read_address_unit = num_wmem
    var.write_address_unit = num_wmem
    var.align = var.read_unit // var.read_address_unit
    var.read_bits = int(math.log2(var.read_unit))
    var.read_address_bits = int(math.log2(var.read_address_unit))
    var.write_address_bits = int(math.log2(var.write_address_unit))
    var.write_bits = int(math.log2(var.write_unit))
    var.read_offset_bits = var.read_bits - var.read_address_bits
    var.read_offset_mask = int(2 ** var.read_offset_bits - 1)
    var.rf_addr_bits = int(math.log2(config.MIDAP.FMEM.NUM_ENTRIES) - var.read_bits)
    var.rf_addr_mask = int(2 ** var.rf_addr_bits - 1)
    var.rw_addr_bits = int(math.log2(config.MIDAP.WMEM.E_NUM_ENTRIES) - var.read_bits)
    var.rw_addr_mask = int(2 ** var.rw_addr_bits - 1)
    var.channel_bits = int(math.log2(config.MIDAP.BMEM.NUM_ENTRIES) - var.write_bits)
    var.channel_mask = int(2 ** var.channel_bits - 1)
    var.fmem_bank_bits = int(math.log2(num_fmem))
    var.wf_addr_bits = var.rf_addr_bits + var.read_bits - var.write_bits
    var.wf_addr_mask = int(2 ** var.wf_addr_bits - 1)
    var.wt_addr_bits = int(math.log2(config.MIDAP.WRITE_BUFFER.NUM_ENTRIES)) - var.write_bits
    var.wt_addr_mask = int(2 ** var.wt_addr_bits - 1)
    var.register_bits = 8 * config.BACKEND.REG_SIZE
    return var

def cim_type(op):
    cim_type = 0
    if isinstance(op, DWConvWrapper) or isinstance(op, MulWrapper) or isinstance(op, SumWrapper):
        cim_type = 4
    elif isinstance(op, AddWrapper):
        cim_type = 5
    elif isinstance(op, AvgpoolWrapper):
        cim_type = 6
    elif isinstance(op, MaxpoolWrapper):
        cim_type = 7
    return cim_type

def act_type(activation):
    from software.network.types import ActivationType
    if activation is None:
        act_type = 0
    elif activation == ActivationType.Linear:
        act_type = 0
    elif activation == ActivationType.ReLU:
        act_type = 1
    else:
        act_type = 2
    return act_type

def bias_type(op):
    bias_type = 0 if op.bias is None else 1
    return bias_type

def get_sram_offset_info(config):
    if "NUM_SLAVES" not in config.DRAM or config.DRAM.NUM_SLAVES == 1:
        SRAM_OFFSET_INFO = AttrDict(
                dict(
                    fmem_pivot_address = 0x60000000,
                    fmem_addr_offset = 0x100000 // config.MIDAP.FMEM.NUM,
                    fmem_id_offset = 7,
                    wmem_pivot_address = 0x60100000,
                    wmem_addr_offset = 0x100000 // (config.MIDAP.WMEM.NUM * 2),
                    wmem_id_offset = 2,
                    bmem_pivot_address = 0x60200000,
                    bmem_addr_offset = 0x10000 // 2,
                    bmem_id_offset = 4,
                    tmem_pivot_address = 0x60210000,
                    tmem_addr_offset = 0x10000 // 2,
                    tmem_id_offset = 0,
                    imem_pivot_address = 0x60220000,
                    imem_addr_offset = 0x10000 // 2,
                    imem_id_offset = (7 + config.MIDAP.FMEM.NUM),
                    lut_pivot_address  = 0x60230000,
                    lut_id_offset = 6,
                )
            )
    elif config.DRAM.NUM_SLAVES == 2:
        SRAM_OFFSET_INFO = AttrDict(
                dict(
                    ## Slave 1
                    fmem_pivot_address = 0x60000000,
                    fmem_addr_offset = 0x100000 // config.MIDAP.FMEM.NUM,
                    fmem_id_offset = 7,
                    lut_pivot_address  = 0x60130000,
                    lut_id_offset = 6,
                    bmem_pivot_address = 0x60100000,
                    bmem_addr_offset = 0x10000 // 2,
                    bmem_id_offset = 4,
                    tmem_pivot_address = 0x60110000,
                    tmem_addr_offset = 0x10000 // 2,
                    tmem_id_offset = 0,
                    imem_pivot_address = 0x60120000,
                    imem_addr_offset = 0x10000 // 2,
                    imem_id_offset = (7 + config.MIDAP.FMEM.NUM),
                    ## Slave 2
                    wmem_pivot_address = 0x60200000,
                    wmem_addr_offset = 0x100000 // (config.MIDAP.WMEM.NUM * 2),
                    wmem_id_offset = 2,
                )
            )
    else:
        raise NotImplementedError()
    return SRAM_OFFSET_INFO

PIPELINE_DEPTH = 15

class TransferType():
    Normal = 0      # static address mapping (deprecated)
    IORead = 1      # input tensor
    IOWrite = 2     # output tensor
    Instruction = 3 # instruction
    Shared = 4      # shared temporal data in a multi-core system (currently NOT used)
    Constant = 5    # weight, bias
    Temporal = 6    # local temporal data of each core

class TriggerType():
    INIT = 0
    MAIN = 1
    RD = 2
    RESET = 3
    DMA_RUN = 1
    SOFTCORE_RUN = 2
