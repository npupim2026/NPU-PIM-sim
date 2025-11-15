import numpy as np
import logging
from config import cfg
from code_generator.env.sfr import reg, SFRSpace
from code_generator.env import rule

from .memory import Memory, Memory3DDMA
from .control_signal_generator import ControlSignalGenerator
from .datapath import Datapath
from .dma_wrapper import DMAWrapper, DMA3DWrapper

class OpcodeSimulator(): # Simualtor & Microblaze role
    def __init__(self):
        self.registers = SFRSpace()
        self.registers.set(reg.sbt, 0)
        self.registers.set(reg.f_sbt, 0)
        self.registers.set(reg.ffc, 0)
        self.registers.set(reg.ffd, 0)
        self.registers.set(reg.ffw, 0)
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('op_sim')
        self.debug = False
        self.hash_func = {
                0x1: self.set,
                0x4: self.wait
                }
    
    def simulation_setup(self, config, midap_manager = None):
        self.manager = midap_manager
        self.svar = rule.make_system_variables(config)
        self.dma_wrapper = DMAWrapper(self, config) if config.DRAM.COMM_TYPE != 'TEST_3D' else DMA3DWrapper(self, config)
        self.control_signal_generator = ControlSignalGenerator(self, config)
        self.datapath = Datapath(self, config)
        self.memory = Memory(config, midap_manager) if config.DRAM.COMM_TYPE != 'TEST_3D' else Memory3DDMA(config, midap_manager)
        self.simulate_trigger = True

    def setup_status(self, request_array = None):
        self.memory.copy_memory()
        self.dma_wrapper.setup_request_array(request_array)
    
    def run_and_compare(self, layer_code, simulate_trigger = True):
        self.simulate_trigger = simulate_trigger
        for i, code in enumerate(layer_code):
            if code.opcode not in self.hash_func:
                raise RuntimeError("Invalid Opcode: {}".format(code))
            self.hash_func[code.opcode](*code.args)
        if isinstance(self.dma_wrapper, DMA3DWrapper):
            self.dma_wrapper.wait_last_transfer()
        if simulate_trigger:
            error = self.memory.compare()
            if error:
                cont = input("Error occurs. Continue? Y/N")
                if cont.lower() not in ['y', 'yes']:
                    raise RuntimeError("Halt Simulation")
    
    def run_datapath(self):
        if not self.simulate_trigger:
            self.logger.warning("Skip Datapath run")
            return
        generator = self.control_signal_generator.generator
        for input_inst in generator:
            self.datapath.run(input_inst)
    
    def wait_datapath(self, cycle):
        generator = self.control_signal_generator.default_generator(cycle)
        for input_inst in generator:
            self.datapath.run(input_inst)
    #Wrapper
    def run_dma(self):
        # transfer-after-write scheme must be implemented for safe transfer..
        self.dma_wrapper.run()

    def set(self, set_value, nop, reg_no, set_type):
        if set_type == 1:
            self.registers.set(reg_no, set_value)
        else:
            raise NotImplementedError("Invalid Opcode: {}".format(set_type))
        if reg_no == reg.trigger_csg.no:
            self.run_datapath()
        elif reg_no in [reg.trigger_dma.no, reg.trigger_wdma.no] and set_value > 0:
            self.run_dma()

    def wait(self, wait_value, nop, reg_no, wait_type):
        if not self.simulate_trigger and reg_no in [reg.trigger_csg, reg.ffd, reg.ffw]:
            return
        if wait_type == 1:
            value = self.registers.get_value(reg_no)
            if value != wait_value:
                self.logger.warning("Expected wait value != reg value, {} vs {} for reg {}".format(
                    wait_value, value, reg_no))
        elif wait_type == 3:
            self.logger.info("Wait for end of the pipeline... {} cycles".format(wait_value))
            self.wait_datapath(wait_value)
        else:
            raise NotImplementedError("Invalid Opcode, wait type: {}".format(wait_type))
