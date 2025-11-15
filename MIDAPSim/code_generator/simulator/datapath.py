import logging
import numpy as np

from config import cfg

from .buffer import InstBuf, DataBuf

#Hardwired Control Logic

class Datapath():
    def __init__(self, simulator, config):
        self.sim = simulator
        self.system_width = config.MIDAP.SYSTEM_WIDTH
        self.num_cims = config.MIDAP.WMEM.NUM
        self.num_slots = self.system_width // self.num_cims
        self.data_buf = DataBuf(config)
        necessary_info = ['phase', 'last']
        write_info = ['wflag', 'wfaddr', 'wtaddr']
        self.debug_buf = InstBuf(svar = self.svar)
        self.input_inst_buf = InstBuf(svar = self.svar)
        self.cim_inst_buf = InstBuf(necessary_info + ['channel_idx', 'reset', 'ignore'] + write_info)
        self.act_inst_buf = InstBuf(necessary_info + ['channel_idx'] + write_info)
        self.reduction_inst_buf = InstBuf(necessary_info + ['channel_idx'] + write_info)
        self.write_inst_buf = InstBuf(necessary_info + write_info)
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('op_sim')
        self.cnt = 0
    
    @property
    def debug(self):
        return self.sim.debug

    @property
    def memory(self):
        return self.sim.memory

    @property
    def registers(self):
        return self.sim.registers

    @property
    def svar(self):
        return self.sim.svar

    @property
    def cc(self): # concurrency
        if (self.registers.cim_type & 0x4) > 0:
            return self.system_width
        else:
            return self.num_cims

    def run(self, instruction):
        if self.debug:
            self.debug_buf.set_from(instruction)
            self.logger.debug("Inst: {}".format(self.debug_buf))
        if instruction.phase == 2:
            self.reduction_inst_buf.set_from(instruction)
            self.reduction_logic()
            self.act_inst_buf.set_from(self.reduction_inst_buf)
            self.activation_logic()
            self.write_inst_buf.set_from(self.act_inst_buf)
            self.write_logic()
        else:
            self.input_inst_buf.set_from(instruction)
            self.input_logic()
            self.cim_inst_buf.set_from(self.input_inst_buf)
            self.cim()
            self.act_inst_buf.set_from(self.cim_inst_buf)
            self.activation_logic()
            self.reduction_inst_buf.set_from(self.act_inst_buf)
            self.write_inst_buf.set_from(self.act_inst_buf)
            self.reduction_logic()
            self.write_logic()
        # else:
        #     self.write_logic()
        #     self.reduction_logic()
        #     self.activation_logic()
        #     self.cim()
        #     self.input_logic()
    
    def input_logic(self): #input controller
        ibuf = self.input_inst_buf
        fbuf = self.data_buf.fbuf
        wbuf = self.data_buf.wbuf
        broadcast_fbuf = self.data_buf.broadcast_fbuf
        ctype = self.registers.cim_type
        if ibuf.phase == 1:
            if ctype < 4:
                fbuf[:self.system_width] = fbuf[self.system_width:]
            self.memory.load_fbuf(fbuf[self.system_width:], ibuf.faddr)
            self.prepare_broadcast(ibuf.align, ibuf.delete)
            if ctype < 6:
                self.memory.load_wbuf(wbuf, ibuf.waddr, 1 if ctype < 4 else 0)
            if self.debug:
                self.logger.debug("Datapath/{}: broadcast_fbuf = {}/{}".format(self.cnt, broadcast_fbuf[:3], np.sum(broadcast_fbuf)))
                self.logger.debug("Datapath/{}: wbuf = {}/{}".format(self.cnt, wbuf[0, :3], np.sum(wbuf[:1, :], axis=1)))
            self.cnt += 1
    
    def prepare_broadcast(self, align, delete):
        fbuf = self.data_buf.fbuf
        broadcast_fbuf = self.data_buf.broadcast_fbuf
        comp = np.left_shift(1, self.num_slots - 1)
        align = self.num_slots - align
        for i in range(self.num_slots):
            broadcast_fbuf[i*self.num_cims:(i+1)*self.num_cims] = \
                    np.zeros(self.num_cims) if (comp & delete) > 0 else \
                    fbuf[(align + i) * self.num_cims:(align + i + 1) * self.num_cims]
            comp = np.right_shift(comp, 1)

    def cim(self): # alu + adder_body
        ibuf = self.cim_inst_buf
        if ibuf.phase == 1:
            self.do_alu()
            self.do_adder(ibuf.reset, ibuf.ignore, ibuf.last)
    
    def do_alu(self):
        ctype = self.registers.cim_type
        wbuf = self.data_buf.wbuf
        broadcast_fbuf = self.data_buf.broadcast_fbuf
        alu_buf = self.data_buf.alu_buf
        activated_cims = self.num_cims if ctype < 4 else 1
        alu_func = np.multiply #0b'000 ~ 0b'100 Conv DWConv Mul
        if ctype == 5: #0b'101 SUM
            alu_func = np.add
        elif ctype > 5: # 0b'110, 111 Avgpool Maxpool
            alu_func = lambda x, y: x
        alu_buf[:activated_cims, :] = alu_func(broadcast_fbuf[:], wbuf[:activated_cims, :])
        if self.debug:
            pass
            #self.logger.debug("Datapath/{}: alu_buf = {}".format(self.cnt, alu_buf[:3, :3]))

    def do_adder(self, reset, ignore, last):
        ctype = self.registers.cim_type
        alu_buf = self.data_buf.alu_buf
        e_output_buf = self.data_buf.e_act_ibuf
        output_buf = self.data_buf.cim_act_ibuf
        if ignore:
            pass
        elif ctype >= 4: # ecim
            ecim_obuf = self.data_buf.ecim_obuf
            if reset:
                ecim_obuf[:] = alu_buf[0, :]
            else:
                if ctype == 7:
                    ecim_obuf[:] = np.maximum(ecim_obuf, alu_buf[0, :])
                else:
                    ecim_obuf[:] = np.add(ecim_obuf, alu_buf[0, :])
            if last:
                e_output_buf[:] = ecim_obuf[:]
            if self.debug:
                pass
                #self.logger.debug("Datapath/{}: adder_buf = {}".format(self.cnt, ecim_obuf[:3]))
        else: # cims 
            cim_obuf = self.data_buf.cim_obuf
            csatree_buf = self.data_buf.csatree_buf
            csatree_buf[:] = np.sum(alu_buf, axis = 1)
            if reset:
                cim_obuf[:] = csatree_buf[:]
            else:
                cim_obuf[:] = np.add(cim_obuf, csatree_buf)
            if last:
                output_buf[:] = cim_obuf[:]
            if self.debug:
                pass
                #self.logger.debug("Datapath/{}: adder_buf = {}".format(self.cnt, cim_obuf[:3]))

    def activation_logic(self):
        ibuf = self.act_inst_buf
        ctype = self.registers.cim_type
        if ibuf.last == 1:
            e_input_buf = self.data_buf.e_act_ibuf
            e_temp_buf = self.data_buf.e_act_tbuf
            c_input_buf = (
                self.data_buf.cim_act_ibuf
                if ibuf.phase != 2
                else self.data_buf.reduction_obuf
            )
            c_temp_buf = self.data_buf.cim_act_tbuf
            a_temp_buf = self.data_buf.activation_tbuf
            output_buf = self.data_buf.activation_obuf
            
            qt = self.registers.quant_type
            bs = self.registers.bias_shift 
            n1 = self.registers.main_shift
            n2 = self.registers.act_shift
            if self.registers.bias_type > 0:
                bbuf = self.data_buf.bias_buf
                self.memory.load_bbuf(bbuf, ibuf.channel_idx)
                bbuf[:] = np.left_shift(bbuf, bs)
                e_temp_buf[:] = np.add(e_input_buf[:], bbuf[:])
                bsel = (ibuf.channel_idx * self.num_cims) % self.system_width
                c_temp_buf[:] = np.add(c_input_buf[:], bbuf[bsel:bsel+self.num_cims])
                if self.debug:
                    self.logger.debug("bbuf = {}".format(bbuf[bsel:bsel+3]))
            else:
                e_temp_buf[:] = e_input_buf[:]
                c_temp_buf[:] = c_input_buf[:]
            if self.debug:
                self.logger.debug("Datapath/{}: wflag = {}, wfaddr = 0x{:0>4X} / wtaddr = 0x{:0>4X}".format(self.cnt, ibuf.wflag, ibuf.wfaddr, ibuf.wtaddr))
                self.logger.debug("e_buf = {}/{}".format(e_temp_buf[:3], np.sum(e_temp_buf)))
                self.logger.debug("c_buf = {}/{}".format(c_temp_buf[:3], np.sum(c_temp_buf)))
            if qt == 1:
                e_temp_buf[:] = np.right_shift(e_temp_buf[:], n1)
                c_temp_buf[:] = np.right_shift(c_temp_buf[:], n1)
            else:
                e_temp_buf[:] = np.left_shift(e_temp_buf[:], n1)
                c_temp_buf[:] = np.left_shift(c_temp_buf[:], n1)
            self.__do_truncate(16, e_temp_buf)
            self.__do_truncate(16, c_temp_buf)
            if self.debug:
                # self.logger.debug("Datapath/{}: wflag = {}, wfaddr = 0x{:0>4X} / wtaddr = 0x{:0>4X}".format(self.cnt, ibuf.wflag, ibuf.wfaddr, ibuf.wtaddr))
                self.logger.debug("e_quant_buf = {}/{}".format(e_temp_buf[:3], np.sum(e_temp_buf)))
                self.logger.debug("c_quant_buf = {}/{}".format(c_temp_buf[:3], np.sum(c_temp_buf)))
            self.__do_activation_func(e_temp_buf, True)
            self.__do_activation_func(c_temp_buf)
            if ctype == 0:
                for i in range(0, self.system_width, self.num_cims):
                    a_temp_buf[i:i+self.num_cims] = c_temp_buf[:]
            else:
                a_temp_buf[:] = e_temp_buf[:]
            a_temp_buf[:] = np.right_shift(a_temp_buf[:], n2)
            self.__do_truncate(8, a_temp_buf)
            output_buf[:] = a_temp_buf[:]
            if self.debug:
                self.logger.debug("output_buf = {}".format(output_buf[:3]))
        self.data_buf.write_buf[:] = self.data_buf.activation_obuf[:]
    
    def __do_truncate(self, n, buf):
        minimum = int(-(2 ** (n-1)))
        maximum = int((2 ** (n-1)) - 1)
        buf[:] = np.where(buf > maximum, maximum, buf)
        buf[:] = np.where(buf < minimum, minimum, buf)
    
    def __do_activation_func(self, buf, ecim = False):
        act_type = self.registers.act_type
        if act_type == 0:
            pass
        elif act_type == 1: # relu
            buf[:] = np.maximum(buf, 0)
        elif act_type == 2 and ecim:
            pass
        elif act_type == 2:
            x1 = np.right_shift(buf, 8)
            x2 = buf - np.left_shift(x1, 8)
            ab = self.memory.get_lut_items(x1)
            y = np.right_shift(ab[:, 0] * x2, 8)
            y = y + ab[:, 1]
            buf[:] = y
            if self.debug:
                # self.logger.debug("Datapath/{}: wflag = {}, wfaddr = 0x{:0>4X} / wtaddr = 0x{:0>4X}".format(self.cnt, ibuf.wflag, ibuf.wfaddr, ibuf.wtaddr))
                self.logger.debug("x1/x2 = {}/{}".format(x1[:3], x2[:3]))
                self.logger.debug("lut_a/lut_b = {}/{}".format(ab[:3,0], ab[:3,1]))
                self.logger.debug("lut_output = {}/{}".format(buf[:3], np.sum(buf)))
        else:
            raise RuntimeError()

    def reduction_logic(self):
        ibuf = self.reduction_inst_buf
        input_buf = self.data_buf.activation_obuf
        reduction_buf = self.data_buf.reduction_buf
        output = self.data_buf.reduction_obuf
        # reduction_dirty_buf = self.data_buf.reduction_dirty_buf
        ci = np.uint32(ibuf.channel_idx) & self.svar.channel_mask
        caddr = ci << self.svar.write_bits
        if ibuf.phase == 3:
            reduction_buf[:] = 0
            if self.debug:
                self.logger.debug("Reduction buf: Reset to 0")
        if self.registers.reduction_type == 1 and ibuf.phase == 1 and ibuf.last:
            # if reduction_dirty_buf[ci] == 1:
            #     reduction_dirty_buf[ci] = 0
            #     reduction_buf[caddr:caddr + self.cc] = input_buf[:self.cc]
            # else:
            reduction_buf[caddr:caddr + self.cc] = np.add(
                    reduction_buf[caddr:caddr + self.cc], input_buf[:self.cc])
            if self.debug:
                self.logger.debug("caddr={}, updated data: {}".format(
                    caddr, reduction_buf[caddr:caddr+3]))
            # self.data_buf.write_buf[:] = self.data_buf.activation_obuf[:]
        elif self.registers.reduction_type == 1 and ibuf.phase == 2 and ibuf.last:
            # if reduction_dirty_buf[ci] == 1:
            #     raise RuntimeError("Dirty output occured in reduction")
            if self.debug:
                self.logger.debug("Reduction Phase... write caddr = {}, data = {}".format(
                    caddr, reduction_buf[caddr:caddr+3]))
            output[:] = reduction_buf[caddr:caddr + self.num_cims]
            # temp_buf[:] = np.right_shift(reduction_buf[caddr:caddr + self.num_cims], n)
            # temp_buf[:] = np.where(temp_buf > 127, 127, temp_buf)
            # temp_buf[:] = np.where(temp_buf < -128, -128, temp_buf)
            # self.data_buf.write_buf[:] = temp_buf
            # reduction_dirty_buf[ci:ci+self.svar.write_bits] = 1
        elif ibuf.phase == 2:
            raise RuntimeError("Reduction type == 0 but phase == 2")
        
    def write_logic(self):
        write_buf = self.data_buf.write_buf
        ibuf = self.write_inst_buf
        cc = self.num_cims if ibuf.phase == 2 else self.cc
        if ibuf.last > 0 and ibuf.wflag > 0:
            self.memory.write(write_buf, ibuf.wflag, ibuf.wfaddr, ibuf.wtaddr, cc)

