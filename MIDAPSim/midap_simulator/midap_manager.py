from __future__ import print_function
from collections import deque

import json
import logging
import logging.config
import numpy as np
import copy
import pickle
import yaml
import math
from data_structure.instruction_components import SLayerInfo

from software.system_compiler.memory_info import MemoryType
from midap_backend.wrapper.op_wrapper import HostProcessWrapper, TestWrapper
from data_structure.attrdict import AttrDict, from_dict_to_attrdict
from config import cfg

from .memory_controller import MemoryController
from .pipeline import get_pipeline
from .statistics import Stats
from .control_logic import get_control_logic

midap_layer_map={}

class MidapManager():

    # MidapManager processes each layer based on given control sequence
    def __init__(self, simulation_level = 0, compile_time = False, *args, **kwargs):
        # initialize submodules (cfg)
        self.stats = Stats()
        self.model_name = ''
        self.simulation_level = simulation_level
        logging_config_dict = cfg.LOGGING_CONFIG_DICT
        if compile_time:
            logging_config_dict['root']['level'] = 'ERROR'
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger()
        self.config = cfg
        self.initialized = False
        self._run_queue = list()
        self.frame_num = -1
        self.midap_layer_map = {}

    @property
    def data_info_dict(self):
        if self.data_info_dict_list is None or self.frame_num < 0:
            return None
        return self.data_info_dict_list[self.frame_num]

    def initialize(self, config = None):
        if config is not None:
            self.config = config
        self.stats.init(config)
        self.data_info_dict_list = None
        self.print_stats = True
        # self.control_logic = get_control_logic(self, self.simulation_level)
        self.memory_controller = MemoryController(self)
        self.control_logic = get_control_logic(self, self.simulation_level)
        self.pipeline = get_pipeline(self, self.simulation_level)
        self.initialized = True
        self.diff_cnt = 0
        self.frame_num = -1

    def simulate(self, simulator_instruction):
        self.setup_simulation(simulator_instruction)
        self._process_network()
    
    def setup_simulation(self, model_name, simulator_instruction):
        self.model_name = model_name
        si = simulator_instruction
        self._setup_simulation(si.config, si.data_info_dict, si.dram_data, si.addr_dict, si.processing_order)

    def simulate_from_file(self, config_file, inst_file, mem_data = None, data_info_file = None):
        self.setup_simulation_from_file(config_file, inst_file, mem_data = None, data_info_file = None)
        for i in range(self.config.MODEL.NUM_FRAMES):
            self.setup_frame(i)
            self._process_network()

    def setup_simulation_from_file(self, config_file, inst_file, mem_data = None, data_info_file = None):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = from_dict_to_attrdict(yaml.load(f, Loader=yaml.SafeLoader))
        if mem_data is None:
            config.DRAM.COMM_TYPE = 'DMA'
        elif config.DRAM.COMM_TYPE == 'DMA':
            config.DRAM.COMM_TYPE = 'VIRTUAL'
        with open(inst_file, 'rb') as f:
            ins = pickle.load(f)
        data_info_dict = None
        if data_info_file is not None:
            with open(data_info_file, 'rb') as f:
                data_info_dict = pickle.load(f)
        self._setup_simulation(config, data_info_dict, mem_data, ins['addr_dict'], ins['processing_order'])

    def _setup_simulation(self, config, data_info_dict, mem_data, addr_dict, path_info):
        if not self.initialized:
            self.initialize(config)
        self.initialized = False
        self.data_info_dict_list = data_info_dict
        self.addr_dict = addr_dict
        self.mem_data = mem_data
        self.path_info = path_info

    def setup_frame(self, frame_num=None):
        if frame_num is None:
            frame_num = self.frame_num + 1
        self.frame_num = frame_num
        self._run_queue = list(self.path_info)
        frame_mem_data = None
        if self.mem_data is not None:
            frame_mem_data = []
            for mem in self.mem_data:
                if len(mem.shape) == 2:
                    frame_mem_data.append(mem[frame_num])
                elif len(mem.shape) == 1:
                    frame_mem_data.append(mem)
        if self.config.DRAM.CLEAR_FRAME_MEM:
            frame_mem_data[MemoryType.Temporal.value][:] = 0
            self.memory_controller.reset_sram()
        self.memory_controller.set_dram_info(frame_mem_data, self.addr_dict)

    def _process_network(self, *args, **kwargs):
        for idx, layer_info in enumerate(self.path_info):
            layer_info2 = self.setup_layer2(layer_info)
            
            if not isinstance(layer_info2.modules[0].op, HostProcessWrapper):
                self.midap_layer_map[layer_info2.name] = idx
        
        file_path = "midap_layer.json"
        reversed_map = {v: k for k, v in self.midap_layer_map.items()}

        with open(file_path, "w") as f:
            json.dump(reversed_map, f, indent=2)

        for idx, layer_info in enumerate(self.path_info):
            self.process_layer(layer_info, idx)
        self.stats.end_simulation()

    def check_and_process(self):
        progress = -1
        if not self._run_queue:
            return 0
        while True:
            if not self._run_queue:
                self.stats.end_simulation()
                return 0
            runnable = self._check_runnable(self._run_queue[0])
            if not runnable:
                return progress
            target = self._run_queue.pop(0)
            self.process_layer(target)
            progress = 1

    def _check_runnable(self, layer_info : SLayerInfo):
        from .shared_info import SyncManager
        wait_b = list(filter(lambda x: x[0] == 'WAIT', layer_info.control_info.behavior_info))
        for b in wait_b:
            wait_id = b[1]
            if wait_id[0] == 0:
                continue
            check = SyncManager.check_sync(wait_id)
            if not check:
                return False
        return True

    def process_layer(self, layer_info, idx):  # layer_idx : debugging info
        self.setup_layer(layer_info)
        if isinstance(self.main_op, HostProcessWrapper):
            self.run_host_proc()
        elif isinstance(self.main_op, TestWrapper):
            self.run_test_proc()
        else:
            self.run()
        self.finish()
        self.logger.info('---------------------------------------------------')
    
    def setup_layer2(self, layer_info : SLayerInfo):
        layer_info2 : SLayerInfo = layer_info
        return layer_info2
    
    def setup_layer(self, layer_info : SLayerInfo):
        self.layer_info : SLayerInfo = layer_info
        self.control_info = layer_info.control_info
        self.main_op = layer_info.modules[0].op
        if not isinstance(self.main_op, HostProcessWrapper):
            self.pipeline.setup(layer_info.modules)
            self.control_logic.setup(layer_info)
            self.memory_controller.setup(layer_info)
            self.on_chip_input_idx = 0
        self.logger.info(str(layer_info))
        return True

    def run_host_proc(self):
        self.logger.info("Host Processing...")
        # Load Input
        # for input_tensor in self.layer_info.input:
        #     if self.config.MIDAP.CONTROL_STRATEGY.FIRST_LAYER != 'EXCLUDE' or input_tensor.name in self.addr_dict:
        #         data_name = input_tensor.name
        #         data_size = np.prod(input_tensor.orig_shape)
        #         self.memory_controller.load_host(data_name, data_size)
        #         self.memory_controller.sync_host()
        # Processing Overhead -> Ignored
        # Write Output
        # for behavior in self.control_info.behavior_info:
        #     btype, i1, i2, i3 = behavior
        #     self.logger.info("Processing: {}, {}, {}:{}".format(btype, i1, i2, i3))
        #     if btype == 'SYNC':
        #         self.memory_controller.sync(i1)
        #     elif btype == 'WAIT':
        #         self.memory_controller.wait_sync(i1)
        # data_name = self.layer_info.modules[0].output[0].name
        # output_data = self.main_op.output_data.reshape(-1)
        # self.memory_controller.write_host(data_name, output_data)

    def run_test_proc(self):
        wait_behav = filter(lambda b: b[0] == 'WAIT', self.control_info.behavior_info)
        sync_behav = filter(lambda b: b[0] == 'SYNC', self.control_info.behavior_info)
        for behavior in wait_behav:
            btype, i1, i2, i3 = behavior
            self.memory_controller.wait_sync(i1)
        for code in self.main_op.test_code:
            action, dependency, param = code
            if action == 'LOAD':
                fmem_idx = dependency
                data_size = param
                data_name = 'input' # FIXME: Temporal solution
                self.memory_controller.memory_manager.load_fmem(
                        fmem_idx,
                        data_name,
                        data_size,
                        0,
                        0,
                        )
                self.stats.read_dram2fmem(data_size)
                self.stats.write_fmem(math.ceil(data_size/self.config.MIDAP.SYSTEM_WIDTH))
            elif action == 'PROCESS':
                memory_delay = 0
                for dep in dependency:
                    if isinstance(dep, int):
                        d = self.memory_controller.memory_manager.access_fmem(dep)
                        memory_delay += d
                        self.stats.wait_dram2fmem(d)
                    elif dep in ['W', 'w']:
                        d = self.memory_controller.memory_manager.access_wmem()
                        memory_delay += d
                        self.stats.wait_dram2wmem(d)
                    elif dep in ['L', 'l']:
                        d = self.memory_controller.memory_manager.access_lut()
                        memory_delay += d
                        self.stats.wait_dram2wmem(d)
                self.stats.increase_cycle(memory_delay + param)
                if self.config.MIDAP.CORE_ID >= 0 and self.config.DRAM.COMM_TYPE == "DMA":
                    self.memory_controller.memory_manager.elapse_cycle()
            elif action == 'WEIGHT_LOAD':
                self.memory_controller.set_next(last_use=False)
            elif action == 'WEIGHT_PREFETCH':
                self.memory_controller.set_next(last_use=True)
        for behavior in sync_behav:
            btype, i1, i2, i3 = behavior
            self.memory_controller.sync(i1)

    def run(self):
        behavior_info = self.control_info.behavior_info
        input_mapping = self.control_info.fmem_info.input_mapping
        for idx, behavior in enumerate(behavior_info):
            btype, i1, i2, i3 = behavior
            self.logger.info("Processing: {}, {}, {}:{}, Cycle:{}".format(btype, i1, i2, i3, self.stats.total_cycle()))
            if btype == 'LOAD':
                cond, data_name, load_idx = i1, i2, i3
                fmem_idx, head, tail = input_mapping[data_name][load_idx]
                self.run_pipeline(cond)
                self.control_logic.sync()
                self.logger.info("[{}/{}]".format(self.stats.current_cycle(), self.stats.total_cycle()))
                self.memory_controller.load_fmem(fmem_idx, data_name, [head, tail], self.layer_info.name)
            elif btype == 'PROCESS':
                process_idx, head_x, tail_x = i1, i2, i3
                last = not any([b[0] == 'PROCESS' for b in behavior_info[idx+1:]])
                self.run_pipeline(-1)
                self.logger.info("[{}/{}]".format(self.stats.current_cycle(), self.stats.total_cycle()))
                self.control_logic.set_generator(head_x, tail_x, self.on_chip_input_idx, behavior.write_info, last)
                self.on_chip_input_idx = process_idx + 1
            elif btype == 'SYNC':
                self.finish_pipeline()
                self.control_logic.sync()
                self.logger.info("SYNC: {}, {}, {}".format(self.stats.total_cycle(), self.stats.current_cycle(), self.stats.dram_delay()))
                self.memory_controller.sync(i1, self.stats.current_cycle() - self.stats.dram_delay(), self.midap_layer_map[self.layer_info.name])
            elif btype == 'WAIT':
                self.memory_controller.wait_sync(i1, self.midap_layer_map[self.layer_info.name])

    def run_pipeline(self, cond):
        for dataflow, simulation_info in self.control_logic.generator:
            running_info, simulated_cycle = simulation_info
            self.stats.increase_cycle(simulated_cycle)
            if self.config.MIDAP.CORE_ID >= 0 and self.config.DRAM.COMM_TYPE == "DMA":
                self.memory_controller.memory_manager.elapse_cycle()
            out_dataflow = self.pipeline.run(dataflow)
            output_phase = out_dataflow.phase
            last_filter, x = running_info.last_filter, running_info.x
            del running_info, out_dataflow
            if output_phase == 3: # Layer processing is finished
                self.logger.info("Layer processing is finished")
                break
            if last_filter and x == cond:
                self.logger.info("Interrupt condition is met")
                break

    def finish(self):
        # Check Functional Result
        # if not isinstance(self.main_op, HostProcessWrapper):
        #     self.finish_pipeline()
        #     self.control_logic.sync()
        if all([
            self.simulation_level == 0,
            self.data_info_dict is not None,
            self.config.MIDAP.FUNCTIONAL_SIM,
            not isinstance(self.main_op, HostProcessWrapper)
            ]
            ):
            self.diff_cnt = {'logical': 0, 'memory': 0}
            for m in self.layer_info.modules:
                for data in m.output:
                    output_mapping = self.control_info.get_output_mapping(data.name)
                    for omap in output_mapping:
                        head_x = max(omap.head, data.offset[0])
                        tail_x = min(omap.tail, data.offset[0] + data.shape[0])
                        if head_x > tail_x:
                            continue
                        head_address = (head_x - omap.head) * output_mapping.yz_plane_size
                        tail_address = (tail_x - omap.head) * output_mapping.yz_plane_size
                        self.data_info_dict[data.name].compare_data_memory[head_x:tail_x, :, :]\
                            = np.reshape(self.memory_controller.memory_manager.fmem[omap.mem_id, head_address:tail_address], (-1, output_mapping.shape[-2], output_mapping.shape[-1]))
                    if output_mapping.write_on_dram_pivot < data.shape[0]:
                        dt, dram_address = self.memory_controller.memory_manager.dram_dict[data.name]
                        self.data_info_dict[data.name].compare_data_memory[output_mapping.write_on_dram_pivot+data.offset[0]:data.shape[0]+data.offset[0], :, :]\
                            = np.reshape(self.memory_controller.memory_manager.dram_data[dt][
                                         dram_address+(output_mapping.write_on_dram_pivot+data.offset[0])*output_mapping.yz_plane_size
                                         : dram_address+(data.offset[0]+data.shape[0])*output_mapping.yz_plane_size],
                                         (-1, output_mapping.shape[-2], output_mapping.shape[-1]))
                    diff_ratio, ret_str = self.data_info_dict[data.name].check_result(data.offset, data.init_shape, m.name)
                    self.logger.info(ret_str)
                    if diff_ratio['logical'] > 0.001 or diff_ratio['memory'] > 0.001:
                        from midap_backend.wrapper.op_wrapper import ArithmeticWrapper
                        self.logger.warning("Diff Ratio > 0.1%... Please check the following layers ) Difference might be caused by zero-padded data")
                        # if not isinstance(self.main_op, ArithmeticWrapper):
                        #     raise RuntimeError
                    self.diff_cnt['logical'] += self.data_info_dict[data.name].diff_cnt['logical']
                    self.diff_cnt['memory'] += self.data_info_dict[data.name].diff_cnt['memory']
            # Check End
        self.stats.set_macs(self.main_op.get_macs())
        self.stats.update(self.layer_info.name, self.print_stats)
    
    def finish_pipeline(self):
        self.run_pipeline(-1)
        self.control_logic.set_finish_generator()
        self.run_pipeline(-1)
        self.stats.increase_cycle(self.pipeline.get_skipped_pipeline_stages())
        if self.config.MIDAP.CORE_ID >= 0 and self.config.DRAM.COMM_TYPE == "DMA":
            self.memory_controller.memory_manager.elapse_cycle()
