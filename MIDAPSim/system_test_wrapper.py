from __future__ import print_function

import copy
import logging
import os
import pickle
from collections import defaultdict, OrderedDict
from functools import reduce
from typing import Any, Callable, List

import numpy as np
import yaml

from config import cfg
from data_structure.attrdict import from_attrdict_to_dict
from data_structure.simulator_instruction import (
    SimulatorInstruction,
    SimulatorInstructionV2,
)
from logger import init_logger
from midap_backend.backend_compiler import BackendCompiler
from midap_backend.wrapper.compile_wrapper import CompileWrapper
from midap_backend.wrapper.op_wrapper import HostProcessWrapper
from midap_simulator import MidapManager
from midap_simulator.shared_info import PrefetchManager, SyncManager
from models import ModelBuilder
from software.compiler.compile_info import CompileInfo
from software.compiler.mapping_compile.compile import MappingCompile
from software.compiler.mapping_compile.policy import (
    LayerPipeline,
    BlockGraphPipeline,
    SingleCoreMappingCompile,
    HostMappingCompile
)
from software.compiler.model_transformer import PruningHeuristic
from software.generic_op import GenericModel
from software.network import ModelGraph
from software.system_compiler.core_info import CoreInfo
from software.system_compiler.system_compiler import SystemCompiler
from software.system_compiler.system_info import SystemInfo
from software.system_compiler.memory_info import MemoryType


class SystemTestWrapper(object):
    def __init__(self, num_cores=1, core_idx=1, simulation_level=0):
        self.generic_model = None
        self.midap_model = None
        self.core_info = [CoreInfo(core_id=i + core_idx) for i in range(num_cores)]
        self.core_idx = core_idx
        self.cm = None
        self.bc = None
        self.si = None
        self.midap_simulator : List[MidapManager] = None
        self.simulation_level = simulation_level
        self.logger = init_logger("TestWrapper", logging.INFO)

    def setup_from_builder(self, builder: ModelBuilder):
        if self.generic_model is not None:
            del self.generic_model
        self.generic_model = GenericModel()
        odict = builder.get_operator_dict()
        self.generic_model.operator_dict = odict
        self.generic_model.post_process()
        self.midap_model = ModelGraph(builder.name)
        self.midap_model.build(self.generic_model.operator_dict)

    def _create_compiler(self, quantize: bool = False):
        from software.compiler.align_compile import AlignCompile
        from software.compiler.block_compile import BlockGraphCompile
        from software.compiler.graph_compile import GraphCompile
        from software.compiler.quant_compile import QuantCompile
        from software.system_compiler.strategy import (
            local_memory_compile,
            shared_memory_compile,
            spm_compile,
            post_process,
        )

        # Pre-Compilation --> Do not affect compile information
        shared_compilers = [QuantCompile, AlignCompile] if quantize else [AlignCompile]
        multicore_compilers = [GraphCompile, MappingCompile]

        gt : str = cfg.MIDAP.CONTROL_STRATEGY.GRAPH_TRANSFORMER
        graph_transformer = PruningHeuristic
        mmp : str = cfg.MIDAP.CONTROL_STRATEGY.MAPPING_POLICY
        multicore_mapping_policy = BlockGraphPipeline # Default
        lmc : str = cfg.MIDAP.CONTROL_STRATEGY.LOCAL_MEMORY_COMPILER
        local_memory_compiler = local_memory_compile.Prototype()
        lsc : str = cfg.MIDAP.CONTROL_STRATEGY.L2_SPM_COMPILER
        l2_spm_compiler = spm_compile.Prototype()
        # Add own Graph Policy Here
        if gt.lower() == "prototype":
            graph_transformer = PruningHeuristic
        # Add own multicore mapping policy here
        if mmp.lower() == "layer_pipe":
            multicore_mapping_policy = LayerPipeline
        elif mmp.lower() == 'prototype':
            multicore_mapping_policy = BlockGraphPipeline
        elif mmp.lower() == 'host':
            multicore_mapping_policy = HostMappingCompile
        # Add own local memory compiler here
        if lmc.lower() == 'prototype':
            local_memory_compiler = local_memory_compile.Prototype
        # Add own l2 spm compiler here
        if lsc.lower() == 'prototype':
            l2_spm_compiler = spm_compile.Prototype
        elif lsc.lower() == 'n2n':
            l2_spm_compiler = spm_compile.N2NStrategy

        if len(self.core_info) > 1:
            multicore_compilers[0].set_transformer(graph_transformer())
        multicore_compilers[1].set_policy(
            SingleCoreMappingCompile(self.cm)
            if len(self.core_info) == 1
            else multicore_mapping_policy(self.cm)
        )
        core_compilers = [BlockGraphCompile]
        offchip_memory_compilers = [
            shared_memory_compile.MultiIOFrame(),
            local_memory_compiler(),
            l2_spm_compiler(),
        ]
        post_process_compilers = [post_process.Prototype()]
        return SystemCompiler(
            shared_compilers,
            multicore_compilers,
            core_compilers,
            offchip_memory_compilers,
            post_process_compilers,
        )

    def compile(
        self, quantize=False, spm_num_banks=2, spm_bank_size=3 * 1024 * 1024, **kwargs
    ):
        if self.cm is not None:
            del self.cm
        kwargs = {}
        cfg.MODEL.QUANTIZED = quantize
        cm = CompileInfo(self.midap_model)
        self.cm = SystemInfo(shared_compile_info=cm, core_info=self.core_info, **kwargs)
        self.cm.memory_info.spm_info.bank_size = spm_bank_size
        self.cm.memory_info.spm_info.num_banks = spm_num_banks
        compiler = self._create_compiler(quantize)
        self.compiler = compiler
        self.cm = compiler.compile(self.cm)
        return self.cm.shared_compile_info

    @staticmethod
    def check_simulation_result(sim: MidapManager) -> str:
        if not sim.config.MIDAP.FUNCTIONAL_SIM or not sim.data_info_dict:
            return 'Not tested'
        elif all([data.diff_cnt == {'logical': 0, 'memory': 0} for data in sim.data_info_dict.values()]):
            return 'Success'
        else:
            return 'Failure'

    def simulate(self, *args, **kwargs):
        self._make_simulation_instruction()
        self.midap_simulator = []
        SyncManager.init()
        PrefetchManager.init_trace(self.cm.generate_prefetch_trace(self.core_idx))
        for idx in range(len(self.core_info)):
            sim_instruction = self.si[idx]
            midap_simulator = MidapManager(self.simulation_level)
            self.midap_simulator.append(midap_simulator)
            midap_simulator.setup_simulation(self.midap_model.name, sim_instruction)
        for i in range(cfg.MODEL.NUM_FRAMES):
            for midap_simulator in self.midap_simulator:
                midap_simulator.setup_frame(i)
            self._system_simulation()
        for idx, midap_simulator in enumerate(self.midap_simulator):
            midap_simulator.stats.print_result(
                self.si[idx].processing_order,
                self.cm.core_info[idx].compile_info.model.name,
            )
            #midap_simulator.memory_controller.memory_manager.show_timeline(
            #    self.si[idx].processing_order, block=(kwargs['sim_last'] and idx == len(self.midap_simulator) - 1))
        return [self.check_simulation_result(midap_simulator) for midap_simulator in self.midap_simulator]

    def _system_simulation(self):
        running_state = [0 for _ in self.midap_simulator]
        while True:
            for idx, midap_simulator in enumerate(self.midap_simulator):
                running_state[idx] = midap_simulator.check_and_process()
            if all([status == -1 for status in running_state]):
                raise RuntimeError("Deadlock occurs...")
            if all([status == 0 for status in running_state]):
                return True


    def run_all(self, model, run=True, save=False, *args, **kwargs):
        self.setup_from_builder(model)
        _ = self.compile(*args, **kwargs)
        self.si = None
        if save:
            self.save_compiled_info(*args, **kwargs)
            return ['Not tested' for i in range(len(self.core_info))]
        if run:
            return self.simulate(*args, **kwargs)

    def _make_simulation_instruction(self):
        if self.cm is None:
            raise RuntimeError("Setup the wrapper first")
        self.si = []
        self.bc = []
        for cinfo in self.cm.core_info:
            core_id = cinfo.core_id
            compile_wrapper = CompileWrapper()
            compile_wrapper.from_core_info(cinfo)  # TODO
            backend_compiler = BackendCompiler(compile_wrapper)
            self.bc.append(backend_compiler)
            backend_compiler.setup_memory(self.cm.get_sim_memory_info(core_id))
            backend_compiler.compile(
                shared_addr_dict=self.cm.get_address_dict(core_id),
            )
            # backend_compiler.setup_shared_memory(mem_info.mem_list)
            # backend_compiler.addr_dict.update(mem_info.address_dict)
            si = SimulatorInstructionV2()
            self.si.append(si)
            si.setup(backend_compiler)

    def save_compiled_info(
        self,
        save_dir="./temp",
        prefix="out",
        data_info=True,
        quantize=False,
        num_frames=1,
        *args,
        **kwargs,
    ):
        import shutil

        if self.si is None:
            self._make_simulation_instruction()
        save_dir = os.path.join(save_dir, prefix)
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        minfo = self.cm.memory_info
        spm_info = minfo.spm_info
        system_config = dict(
            core_ids=[core.core_id for core in self.core_info],
            spm_config=dict(num_banks=spm_info.num_banks, bank_size=spm_info.bank_size),
            quantized=quantize,
            num_frames=num_frames,
        )
        system_config_file = os.path.join(save_dir, "system_config.yml")
        with open(system_config_file, "w", encoding="utf-8") as f:
            yaml.dump(system_config, f, default_flow_style=False, encoding="utf-8")
        data_shared_feature_file = os.path.join(save_dir, "data_shared.bin")
        data_shared_input_file = os.path.join(save_dir, "data_in.bin")
        data_shared_output_file = os.path.join(save_dir, "data_out.bin")
        data_shared_constants_file = os.path.join(save_dir, "data_const.bin")
        minfo.data_shared_feature.tofile(data_shared_feature_file)
        minfo.data_input_feature.tofile(data_shared_input_file)
        minfo.data_output_feature.tofile(data_shared_output_file)
        minfo.data_shared_constants.tofile(data_shared_constants_file)
        if spm_info.in_use:
            offsets = {
                0: cfg.DRAM.OFFSET.SHARED,
                1: cfg.DRAM.OFFSET.INPUT,
                2: cfg.DRAM.OFFSET.OUTPUT,
                3: cfg.DRAM.OFFSET.WEIGHT_BIAS,
                4: cfg.DRAM.OFFSET.BUFFER,
            }
            traces = self.cm.generate_prefetch_trace(self.core_idx)
            for idx, bank_trace in traces.items():
                prefetch_file = os.path.join(save_dir, f"prefetch_bank_{idx}.txt")
                with open(prefetch_file, "w") as f:
                    for trace in bank_trace:
                        f.write(trace.to_str(offsets))
            prefetch_dump_file = os.path.join(save_dir, "prefetch.bin")
            with open(prefetch_dump_file, "wb") as f:
                pickle.dump(traces, f)
        core_info: CoreInfo
        si: SimulatorInstruction
        layers = reduce(lambda x, y: {**x, **y}, (si.compile_info.layers for si in self.si), OrderedDict())
        for core_info, si in zip(self.core_info, self.si):
            target_dir = os.path.join(save_dir, f"core_{core_info.core_id}")
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            if not spm_info.in_use:
                buf_file = os.path.join(target_dir, "data_buf.bin")
                self.logger.info("Store dram to {} " + buf_file)
                si.dram_data[MemoryType.Temporal.value].tofile(buf_file)
            if data_info:
                data_info_file = os.path.join(target_dir, "gt_info.pkl")
                self.logger.info("Store data_info_dict file to {}" + data_info_file)
                with open(data_info_file, "wb") as f:
                    pickle.dump(si.data_info_dict, f)
            inst = dict(addr_dict=si.addr_dict, processing_order=si.processing_order)
            inst_file = os.path.join(target_dir, "inst.pkl")
            self.logger.info("Store instruction file to {}" + inst_file)
            with open(inst_file, "wb") as f:
                pickle.dump(inst, f)
            config_file = os.path.join(target_dir, "config.yml")
            self.logger.info("Store config file to {}" + config_file)
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    from_attrdict_to_dict(copy.copy(si.config)),
                    f,
                    default_flow_style=False,
                    encoding="utf-8",
                )
            self.logger.error(f"core {core_info.core_id} Save Finished.")
            host_wait_info = defaultdict(list)
            host_sync_info = defaultdict(list)
            sync2layer = {(core_id, layer_id): layer for core_id in self.cm.dependency_info.layer2id for layer, layer_id in self.cm.dependency_info.layer2id[core_id].items()}
            for layer_info in filter(lambda l: isinstance(l.modules[0].op, HostProcessWrapper), si.processing_order):
                for behavior in layer_info.control_info.behavior_info:
                    btype, i1, i2, i3 = behavior
                    if btype == 'WAIT':
                        host_mapping = layer_info.modules[0].op.mapping
                        target_layer = sync2layer[i1]
                        output_tensor = layers[target_layer].main_output
                        host_wait_info[host_mapping].append((i1, si.addr_dict[output_tensor.name], np.prod(output_tensor.shape), layer_info.name))
                    elif btype == 'SYNC':
                        host_mapping = layer_info.modules[0].op.mapping
                        target_layer = sync2layer[i1]
                        output_tensor = layers[target_layer].main_output
                        host_sync_info[host_mapping].append((i1, si.addr_dict[output_tensor.name], np.prod(output_tensor.shape), layer_info.name))
            for mapping in set(host_wait_info.keys()).union(set(host_sync_info.keys())):
                host_wait_file = os.path.join(target_dir, f"{mapping}_wait_info.txt")
                host_sync_file = os.path.join(target_dir, f"{mapping}_sync_info.txt")
                with open(host_wait_file, 'w') as f:
                    for (id, addr, size, layer_name) in host_wait_info[mapping]:
                        f.write(f'{id[1]},{addr[0]},{addr[1]},{size},{layer_name}\n')
                with open(host_sync_file, 'w') as f:
                    for (id, addr, size, layer_name) in host_sync_info[mapping]:
                        f.write(f'{id[1]},{addr[0]},{addr[1]},{size},{layer_name}\n')
        return None

    @staticmethod
    def __prepare_running_info(load_dir : str):
        system_config_file = os.path.join(load_dir, "system_config.yml")
        with open(system_config_file, "r", encoding="utf-8") as f:
            system_config = yaml.load(f)
        mem_postfix = ["data_shared.bin", "data_in.bin", "data_out.bin", "data_const.bin"]
        dtype = np.int8 if system_config["quantized"] else np.float32
        mem_data = []
        for postfix in mem_postfix:
            mem_file = os.path.join(load_dir, postfix)
            with open(mem_file, "rb") as f:
                mem_data.append(np.fromfile(f, dtype=dtype))
        mem_data[MemoryType.Input.value] = mem_data[MemoryType.Input.value].reshape(system_config["num_frames"], -1)
        spm_size = (
            system_config["spm_config"]["bank_size"]
            * system_config["spm_config"]["num_banks"]
        )
        traces = {}
        if spm_size > 0:
            mem_data.append(np.zeros(spm_size, dtype=dtype))
            prefetch_dump_file = os.path.join(load_dir, "prefetch.bin")
            with open(prefetch_dump_file, "rb") as f:
                traces = pickle.load(f)
        PrefetchManager.init_trace(traces)
        SyncManager.init()        
        return system_config, mem_data
    
    @staticmethod
    def run_system_func(load_dir: str, func : Callable[[Any], Any]):
        system_config, mem_data = SystemTestWrapper.__prepare_running_info(load_dir)
        dtype = np.int8 if system_config["quantized"] else np.float32
        spm_size = (
            system_config["spm_config"]["bank_size"]
            * system_config["spm_config"]["num_banks"]
        )
        for core_id in system_config["core_ids"]:
            core_dir = os.path.join(load_dir, f"core_{core_id}")
            core_mem_data = mem_data
            if spm_size == 0:  # Use temporal data
                core_mem_file = os.path.join(core_dir, "data_buf.bin")
                with open(core_mem_file, "rb") as f:
                    core_mem_data = mem_data + [np.fromfile(f, dtype=dtype)]
            func(core_dir, core_mem_data)


    def run_system_from_dir(self, load_dir="./temp", prefix="out", *args, **kwargs):
        load_dir = os.path.join(load_dir, prefix)
        self.midap_simulator = []
        func = lambda core_dir, core_mem_data: self.setup_core_from_dir(core_dir, core_mem_data, *args, **kwargs)
        self.run_system_func(load_dir, func)
        cfg.MODEL.NUM_FRAMES = max([midap_simulator.config.MODEL.NUM_FRAMES for midap_simulator in self.midap_simulator])
        for i in range(cfg.MODEL.NUM_FRAMES):
            for midap_simulator in self.midap_simulator:
                midap_simulator.setup_frame(i)
            self._system_simulation()
        for idx, midap_simulator in enumerate(self.midap_simulator):
            midap_simulator.stats.print_result(
                midap_simulator.path_info,
                f"{prefix}_core_{idx}"
            )
            #midap_simulator.memory_controller.memory_manager.show_timeline(
            #    midap_simulator.path_info, block=(idx == len(self.midap_simulator) - 1))
        return [self.check_simulation_result(midap_simulator) for midap_simulator in self.midap_simulator]


    def setup_core_from_dir(
        self, core_dir: str, mem_data=None, data_info=True, *args, **kwargs
    ):
        if self.midap_simulator is None:
            self.midap_simulator = []
        sim = MidapManager(self.simulation_level)
        self.midap_simulator.append(sim)
        config_file = os.path.join(core_dir, "config.yml")
        inst_file = os.path.join(core_dir, "inst.pkl")
        data_info_file = None
        if data_info:
            data_info_file = os.path.join(core_dir, "gt_info.pkl")
        sim.setup_simulation_from_file(config_file, inst_file, mem_data, data_info_file)

    def run_core_from_dir(
        self, core_dir: str, mem_data=None, data_info=True, *args, **kwargs
    ):
        if self.midap_simulator is None:
            self.midap_simulator = []
        sim = MidapManager(self.simulation_level)
        self.midap_simulator.append(sim)
        config_file = os.path.join(core_dir, "config.yml")
        inst_file = os.path.join(core_dir, "inst.pkl")
        data_info_file = None
        if data_info:
            data_info_file = os.path.join(core_dir, "gt_info.pkl")
        sim.simulate_from_file(config_file, inst_file, mem_data, data_info_file)
        po = sim.path_info
        sim.stats.print_result(
            po,
            "run_from_file",
        )
        #sim.memory_controller.memory_manager.show_timeline(po, block=False)
        return [self.check_simulation_result(sim)]

