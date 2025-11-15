from code_generator.simulator.tvm import TVM
import copy
import numpy as np
import logging

from config import cfg

from code_generator.env import rule
from code_generator.env.sfr import reg
from code_generator.env.rule import TriggerType
from .address_module import AddressModule
from .buffer import InstBuf


class ControlSignalGenerator():
    def __init__(self, simulator, config):
        self.sim = simulator
        self.fmem_address_module = AddressModule(6) # (pivot+s), y, z, i, j, k
        self.wmem_address_module = AddressModule(6) # (pivot+s), y, z, i, j, k
        self.channel_address_module = AddressModule(2) # (pivot, z)
        self.out_fmem_address_module = AddressModule(3) # (pivot, y, z)
        self.out_tmem_address_module = AddressModule(3) # (pivot, y, z)
        self.tvm = TVM()
        self.input_inst_buf = InstBuf() 
        #self.write_logic = WriteLogic(self, memory)
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('op_sim')
        self.generator_dict = {
            TriggerType.INIT: lambda x: self.default_generator(1),
            TriggerType.MAIN: self.generic_generator,
            TriggerType.RD: self.generic_generator,
            TriggerType.RESET: self.reset_generator,
            }
        self.sram_offset_info = rule.get_sram_offset_info(config)
    
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
    def fam(self):
        return self.fmem_address_module

    @property
    def wam(self):
        return self.wmem_address_module

    @property
    def cam(self):
        return self.channel_address_module

    @property
    def ofam(self):
        return self.out_fmem_address_module

    @property
    def otam(self):
        return self.out_tmem_address_module

    @property
    def wflag(self):
        return self.registers.get_value(reg.wi)[0]
    
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

    def set_inst_buf(
            self,
            phase = 0,
            delete = 0,
            align = 0,
            reset = 0,
            ignore = 0,
            last = 0,
            ):
        faddr = int(self.fam.address) >> self.svar.read_offset_bits
        waddr = int(self.wam.address) >> self.svar.read_offset_bits
        channel_idx = self.cam.address
        wflag = self.wflag
        ### Actually Dont'care, but insert it to avoid overflow
        wfaddr = 0 if (wflag & 0x2) == 0 else self.ofam.address
        wtaddr = 0 if (wflag & 0x1) == 0 else self.otam.address 
        if phase != 1:
            waddr = 0
            faddr = 0
        setup_dict = dict(
                phase = phase,
                faddr = faddr,
                waddr = waddr,
                channel_idx = channel_idx,
                delete = delete,
                align = align,
                reset = reset,
                ignore = ignore,
                last = last,
                wflag = wflag,
                wfaddr = wfaddr,
                wtaddr = wtaddr
                )
        self.input_inst_buf.set_from(setup_dict)

    @property
    def generator(self):
        trigger, trigger_id = self.registers.get_value(reg.trigger_csg)
        if trigger not in self.generator_dict:
            raise ValueError("Triggering {}: invalid signal".formant(trigger))
        else:
            self.logger.debug("Triggering Control Signal Generator: {}, id {}".format(trigger, trigger_id))
        self.check_access_and_set(trigger)
        self.registers.set(reg.feedback_from_csg, trigger_id)
        self.tvm.set(self.registers.get_value(reg.tvi1) + self.registers.get_value(reg.tvi2) + self.registers.get_value(reg.rpad))
        return iter(self.generator_dict[trigger](trigger))

    def default_generator(self, cycle):
        for _ in range(cycle): # For sequential running
            self.set_inst_buf()
            yield self.input_inst_buf
    
    def check_access_and_set(self, trigger):
        if trigger in [TriggerType.RESET, TriggerType.INIT]:
            return
        busy_table = self.registers.get_value(reg.sbt)
        check = self._make_check_bin(trigger, init = True)
        if check & busy_table > 0:
            self.logger.warning("Check: {}, busy_table: {}, Read/Write Conflict: ".format(
                np.binary_repr(check, width = 10), np.binary_repr(busy_table, width=10)))
        busy_table = check | busy_table
        self.registers.set(reg.f_sbt, busy_table)
    
    def disable_busy_table(self, trigger):
        self.registers.set(reg.f_sbt, 0)

    def _make_check_bin(self, trigger, init = False):
        rfp, rwp = self.registers.get_value(reg.rp)
        jv, _, jfp = self.registers.get_value(reg.ji)
        wflag, wtp, wfp = self.registers.get_value(reg.wi)
        cp, _ = self.registers.get_value(reg.bi)
        fcheck = 0
        jfcheck = 0
        wcheck = 0
        bcheck = 0
        tcheck = 0
        wfcheck = 0 # Must not be conflicted
        if trigger == TriggerType.MAIN:
            fcheck = 1
            if self.registers.cim_type < 6:
                wcheck = 1
            if jv > 0:
                jfcheck = 1
            if self.registers.bias_type > 0:
                bcheck = 1
        if wflag & 0x2 > 0:
            wfcheck = 1
        if wflag & 0x1 > 0:
            tcheck = 1
        fp_bits = self.fid_offset + (rfp >> (self.svar.rf_addr_bits + self.svar.read_offset_bits))
        jp_bits = self.fid_offset + (jfp >> (self.svar.rf_addr_bits + self.svar.read_offset_bits))
        wp_bits = self.wid_offset + (rwp >> (self.svar.rw_addr_bits + self.svar.read_offset_bits))
        cp_bits = self.bid_offset + (cp >> self.svar.channel_bits)
        wtp_bits = self.tid_offset + (wtp >> self.svar.wt_addr_bits)
        wfp_bits = self.tid_offset + (wfp >> self.svar.wf_addr_bits)
        fcheck = fcheck << fp_bits
        jfcheck = jfcheck << jp_bits
        wcheck = wcheck << wp_bits
        bcheck = bcheck << cp_bits
        tcheck = tcheck << wtp_bits
        wfcheck = wfcheck << wfp_bits
        check = fcheck | jfcheck | wcheck | bcheck | tcheck | wfcheck
        return check
    
    def process_address_signal(self, signal, sel = 0, fignore = 0, wignore = 0):
        assert len(signal) == 6
        # read info
        rfp, rwp = self.registers.get_value(reg.rp)
        jv, jx, jfp = self.registers.get_value(reg.ji)
        fso, wso = self.registers.get_value(reg.rso)
        fyo, wyo = self.registers.get_value(reg.ro1)
        fzo, wzo = self.registers.get_value(reg.ro2)
        fio, wio = self.registers.get_value(reg.ro3)
        fjo, wjo, fko, wko = self.registers.get_value(reg.ro4)
        rwp = rwp + wso
        # setup read address modules
        fam_signal = copy.copy(signal)
        if fignore == 1:
            fam_signal[-1] = 0
        if sel == 1:
            fam_signal[-3] = 3
            fam_signal[0] = 1
            rfp = jfp
        rfp += fso
        if self.debug and self.tvm.activate > 0:
            self.logger.debug(f"FAM Signal: {fam_signal}, TVM Status: {self.tvm}")
        fam_signal = self.tvm.process_signal(sel, fam_signal)
        if self.debug and self.tvm.activate > 0:
            self.logger.debug(f"=-----> Processed FAM Signal: {fam_signal}, TVM Status: {self.tvm}")
        self.fam.set(fam_signal, [rfp, fyo, fzo, fio, fjo, fko])
        wam_signal = copy.copy(signal)
        if wignore == 1:
            wam_signal[-1] = 0
        self.wam.set(wam_signal, [rwp, wyo, wzo, wio, wjo, wko])
        # write info
        wflag, wtp, wfp = self.registers.get_value(reg.wi)
        wfyo, wfzo = self.registers.get_value(reg.wo1)
        wtyo, wtzo = self.registers.get_value(reg.wo2), wfzo
        cp, czo = self.registers.get_value(reg.bi)
        # setup output address modules
        self.cam.set([signal[0], signal[2]], [cp, czo])
        self.ofam.set(signal[:3], [wfp, wfyo, wfzo])
        self.otam.set(signal[:3], [wtp, wtyo, wtzo])
    
    def generic_generator(self, trigger):
        jv, jx, _ = self.registers.get_value(reg.ji)
        y_iter, z_iter = self.registers.get_value(reg.oli)
        kx, ky, kz = self.registers.get_value(reg.ili)
        wo, _fo, jfo, fio, yo, yzo, delete_f, delete_b = self.registers.get_value(reg.cyzi)
        if self.debug:
            self.logger.debug("ConvZY Called: iter(y,z,kx,ky,kz) = {}".format((y_iter,z_iter,kx,ky,kz)))
            self.logger.debug("ConvZY Params: wo, _fo, jfo, fio, yo, yzo = {}".format([
                wo, _fo, jfo, fio, yo, yzo]))
            self.logger.debug("Pre-generated del offset: (f, b) = {}".format([delete_f, delete_b]))
            self.logger.debug(f"TVM parameters: {self.tvm.params_str}")
        if jv > 0:
            self.logger.debug("Jump x: {}".format(jx))
        valid_jump = jv > 0
        signal = [1, 3, 3, 3, 3, 3] ## int state --> state transition
        sel = 0
        sig_ignore = 0
        offset = 0
        phase = 1 if trigger != TriggerType.RD else 2
        #eff_last = wo + yzo
        #delete_f = np.left_shift(np.left_shift(1, wo) - 1, 4 - wo) # Fixed .. can be gotten from register
        #delete_b = (np.left_shift(1, 4 - eff_last) - 1) # Fixed.. can be gotten from register
        for oy in range(y_iter + 1):
            for oz in range(z_iter + 1):
                reset = 1
                last = 0
                fop = _fo
                for i in range(kx + 1):
                    fo = (fop + offset) & self.svar.read_offset_mask ## In rtl, it can be solved by using 2-bits adder
                    if self.debug and oz == 0:
                        self.logger.debug("foffset: (oy, oz, i) / (fop, offset, fo, wo) = {}, {}".format(
                            [oy, oz, i],[fop, offset, fo, wo]))
                    if wo < fo:
                        align = wo - fo  + (2 ** self.svar.read_offset_bits) #### In rtl, it can be solved by using 2-bits subtractor
                        ignore = 1
                    else:
                        align = wo - fo
                        ignore = 0
                    fignore = -(fo + yzo) % 2 ** self.svar.read_offset_bits < align
                    new_kz = kz + ignore
                    first = 1
                    for j in range(ky + 1): # ky == 0
                        for k in range(new_kz + 1):
                            sig_fignore = 0
                            last_mask = 0x0
                            first_mask = 0x0
                            if first == 1:
                                first_mask = 0xF
                            if j == ky and k == new_kz:
                                last_mask = 0xF
                                if not first == 1:
                                    sig_fignore = fignore
                            delete = (first_mask & delete_f) | (last_mask & delete_b)
                            self.process_address_signal(signal, sel=sel, fignore = sig_fignore, wignore = sig_ignore) # 
                            if i == kx and j == ky and k == new_kz:
                                last = 1
                            self.set_inst_buf(
                                phase = phase,
                                align = align,
                                delete = delete,
                                reset = reset,
                                ignore = ignore,
                                last = last,
                                )
                            yield self.input_inst_buf
                            reset = reset & ignore
                            first = ignore
                            signal = [0, 0, 0, 0, 0, 2]
                            sig_ignore = ignore
                            delete = 0
                            ignore = 0
                            sel = 0
                        signal[5] = 3
                        signal[4] = 2
                    signal[4] = 3
                    signal[3] = 2
                    fop += fio
                    if valid_jump and i == jx:
                        sel = 1
                        fop = jfo
                signal[3] = 3
                signal[2] = 2
                signal[0] = 1
                sel = 0
            offset += yo
            signal[2] = 3
            signal[1] = 2
        ### Out state --> init state transition
        self.process_address_signal(signal)
        self.disable_busy_table(trigger)

    def reset_generator(self, trigger = 0):
        self.process_address_signal([1, 3, 3, 3, 3, 3]) # 
        self.set_inst_buf(phase = TriggerType.RESET)
        yield self.input_inst_buf

