from enum import Enum, auto
import numpy as np
from data_structure.attrdict import AttrDict

from .sfr import reg

class Opcode():
    def __init__(self, *args, **kwargs):
        self.name = None
        self.opcode = None # int, 5bit
        self.args = None # int list, LEFT To RIGHT (Higher bit -> Lower bit order)
        self.arg_bits = None # int list: bits per each args, sum(self.arg_bits) == 27
        self.setup(*args, **kwargs)
        assert self.opcode < 0x20
        assert any([isinstance(self.args, list), isinstance(self.args, tuple)]) and any([isinstance(self.arg_bits, list), isinstance(self.arg_bits, tuple)])
        assert sum(self.arg_bits) == 59
        assert len(self.arg_bits) == len(self.args)
    
    def setup(self, **kwargs):
        self.name = 'NOP'
        self.opcode = 0x0
        self.args = [0x0] 
        self.arg_bits = [59]

    def to_binary_str(self):
        opcode_str = np.binary_repr(self.opcode, width=5)
        parameter_str = ''
        for arg, bits in zip(self.args, self.arg_bits):
            parameter_str += np.binary_repr(arg, width=bits)
        return parameter_str + opcode_str

    def to_hex_str(self):
        binary_str = self.to_binary_str()
        hex_str = format(int(binary_str, 2), 'x')
        hex_str_fragment = [hex_str[max(0, i-4):i] for i in range(len(hex_str), 0, -4)]
        return '_'.join(reversed(hex_str_fragment))

    def to_binary(self):
        binary_str = self.to_binary_str()
        byte_arr = bytearray([int(binary_str[56-8*i:64-8*i], 2) for i in range(8)])
        return byte_arr

    def __repr__(self):
        return self.name

class SET(Opcode):
    #set_type: b'001 = set, b'100 = set reg[value:value] bits true, b'110 = set reg[value:value] bits false
    def setup(self, value, reg_id, set_type = 0, *args, **kwargs):
        assert value < 0x100000000 and reg_id < 0x100 and set_type < 0x8
        self.name = 'SET'
        self.opcode = 0x1
        self.args = [value, 0, reg_id, set_type]
        self.arg_bits = [32, 16, 8, 3]

    def __repr__(self):
        ret = "SET - REG:{:x} VALUE:{:x} TYPE:{:x} CODE:".format(self.args[2], self.args[0], self.args[3]) + self.to_hex_str()
        return ret

class WAIT(Opcode):
    def setup(self, value, reg_id, wait_type = 0, *args, **kwargs):
        assert value < 0x100000000 and reg_id < 0x100 and wait_type < 0x8
        self.name = 'WAIT'
        self.opcode = 0x4
        self.args = [value, 0, reg_id, wait_type]
        self.arg_bits = [32, 16, 8, 3]

    def __repr__(self):
        ret = "WAIT - REG:{:x} VALUE:{:x} TYPE:{:x} CODE:".format(self.args[2], self.args[0], self.args[3]) + self.to_hex_str()
        return ret

def set_reg(reg_id, value):
    assert value >= 0 and value < 0x100000000
    return [SET(value, reg_id, 1)]

def reset_reg(reg):
    reg_no = reg if isinstance(reg, int) else reg.no
    return [SET(0, reg_no, 1)]

# def bitset(reg_id, bit, flag = True):
#     set_type = 0x4 if flag else 0x6
#     return [SET(bit, reg_id, set_type)]

def bitwait(reg_id, bit, flag = False):
    wait_type = 0x4 if flag else 0x6
    return [SET(bit, reg_id, wait_type)]

def wait_reg_zero(reg_id):
    return wait_reg_eq(reg_id, 0)

def wait_reg_eq(reg_id, value):
    return wait_reg(reg_id, value, 1)

def wait_reg(reg_id, value, wait_type):
    assert value >= 0 and value < 0x100000000
    return [WAIT(value, reg_id, wait_type)]

def wait_fin(cycle): # Wait pipeline size cycle
    assert cycle >= 0 and cycle < 0x10000
    return [WAIT(cycle, 0, 0x3)]
