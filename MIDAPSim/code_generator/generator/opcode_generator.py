from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import logging.config
import traceback
import numpy as np
import copy
import os
import os.path
import yaml
import pickle

from midap_backend.wrapper.op_wrapper import HostProcessWrapper
from data_structure.attrdict import AttrDict, from_dict_to_attrdict
from config import cfg

from code_generator.env.sfr import reg, SFRSpace
from code_generator.env import rule

from .control_op_generator import ControlOpGeneratorLv0
from .memory_op_generator import MemoryOpGenerator

class OpcodeGenerator():
    def __init__(self, *args, **kwargs):
        self.control_op_generator = None
        self.memory_op_generator = None
        self.opcode_simulator = None
        self.registers = None
        self.svar = None
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('gen')
        self.initialized = False
        self.info = [None, None, None, None, None] # config, processing order, addr_dict, dram_data, data_info_dict
    
    @property
    def config(self):
        return self.info[0]

    def setup_from_file(self, config_file, inst_file, dram_data, data_info_file = None, dynamic_base_addr = False, *args, **kwargs):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = from_dict_to_attrdict(yaml.load(f))
        with open(inst_file, 'rb') as f:
            ins = pickle.load(f)
        data_info_dict = None
        if data_info_file is not None:
            with open(data_info_file, 'rb') as f:
                data_info_dict = pickle.load(f)
        info = [config, ins['processing_order'], ins['addr_dict'], dram_data, data_info_dict]
        self.info = info
        self.dynamic_base_addr = dynamic_base_addr
        self.initialized = False
    
    def setup_from_instruction(self, simulator_instruction, dynamic_base_addr, *args, **kwargs):
        si = simulator_instruction
        info = [si.config, si.processing_order, si.addr_dict, si.dram_data, si.data_info_dict]
        self.info = info
        self.dynamic_base_addr = dynamic_base_addr
        self.initialized = False

    def generate_layer_code(self, layer_info):
        if isinstance(layer_info.modules[0].op, HostProcessWrapper):
            return []
        self.logger.info("------------------------------------------")
        self.logger.info("Current layer: {}".format(layer_info))
        setup_code = self.setup_layer(layer_info)
        main_code = self._gen_main_code(layer_info)
        fin_code = self._gen_fin_code(layer_info)
        return setup_code + main_code + fin_code
    
    def initialize(self):
        self.initialized = True
        self.svar = rule.make_system_variables(self.config)
        self.registers = SFRSpace(num_sfrs = 80, dtype = np.int32)
        self.memory_op_generator = MemoryOpGenerator(self)
        self.control_op_generator = ControlOpGeneratorLv0(self)
        self.memory_op_generator.set_dram_info(self.info[2])
        init_code = self.memory_op_generator.gen_init_code()
        init_code += self.control_op_generator.gen_init_code()
        return init_code

    def setup_layer(self, layer_info):
        init_code = []
        if not self.initialized:
            init_code = self.initialize()
        else:
            init_code = self.control_op_generator.gen_init_code()
        self.on_chip_input_idx = 0
        sync_code = []
        setup_memory_code = []
        setup_cg_code = []
        # if layer_info.control_info.behavior_info.require_sync:
        #     sync_code = self.memory_op_generator.sync()
        if not isinstance(layer_info.modules[0].op, HostProcessWrapper):    # Currently always true
            setup_memory_code = self.memory_op_generator.setup(layer_info)
            setup_cg_code = self.control_op_generator.setup(layer_info)
        return init_code + sync_code + setup_memory_code + setup_cg_code

    def _gen_main_code(self, layer_info):
        control_info = layer_info.control_info
        behavior_info = control_info.behavior_info
        input_mapping = control_info.fmem_info.input_mapping
        main_code = []
        for idx, behavior in enumerate(behavior_info):
            btype, i1, i2, i3 = behavior
            self.logger.info("Processing: {}, {}, {}:{}".format(btype, i1, i2, i3))
            if btype == 'LOAD':
                cond, data_name, load_idx = i1, i2, i3
                fmem_idx, head, tail = input_mapping[data_name][load_idx]
                c1, c2 = self.run_generator(cond)
                main_code += c1
                main_code += self.memory_op_generator.load_fmem(fmem_idx, data_name, [head, tail])
                main_code += c2
            elif btype == 'PROCESS':
                process_idx, head_x, tail_x = i1, i2, i3
                last = not any([b[0] == 'PROCESS' for b in behavior_info[idx+1:]])
                main_code += self.run_generator(-1)[0]
                self.control_op_generator.set_generator(head_x, tail_x, self.on_chip_input_idx, behavior.write_info, last)
                self.on_chip_input_idx = process_idx + 1
        return main_code
    
    def run_generator(self, cond):
        gen_code = []
        for code, running_info in self.control_op_generator.generator:
            x, last_filter = running_info
            if last_filter and x == cond:
                self.logger.info("Interrupt condition is met")
                break
            gen_code += code
            code = []
        return gen_code, code

    def _gen_fin_code(self, *args, **kwargs):
        code, _ = self.run_generator(-1)
        self.control_op_generator.set_finish_generator()
        code += self.run_generator(-1)[0]
        code += self.memory_op_generator.gen_fin_code()
        return code
