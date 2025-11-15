from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import logging.config
import traceback
import copy
import os
import os.path
import numpy as np

from midap_simulator import MidapManager
from config import cfg

from .simulator import OpcodeSimulator
from .generator import OpcodeGenerator
from .generator.memory_op_generator import TransferInfo, RDMA3DDescriptor, WDMA3DDescriptor
from .env import rule
from software.system_compiler.memory_info import MemoryType
from midap_backend.wrapper.op_wrapper import HostProcessWrapper

class OpcodeTool(object):
    def __init__(self):
        self.manager = None
        self.code_generator = None
        self.code_simulator = None
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger()

    def setup_generator(self, from_file = False, *args, **kwargs):
        self.code_generator = OpcodeGenerator()
        if not from_file:
            self.code_generator.setup_from_instruction(*args, **kwargs)
        else:
            self.code_generator.setup_from_file(*args, **kwargs)

    def run(self, verify = False, *args, **kwargs):
        if verify:
            self._verify(*args, **kwargs) # For test only
        else:
            self._generate_bin(*args, **kwargs) # For binary generation

    def _generate_bin(
            self,
            integrate = False,
            out_dir_name = './microcode',
            out_prefix = 'inst',
            postfix = '.bin',
            generate_gold_data = False,
            *args,
            **kwargs,
            ):
        os.makedirs(out_dir_name, exist_ok=True)
        zero_pad = bytearray(8)
        addr = 0
        skipped_layer = 0
        config, processing_order, dram_dict, dram_info, data_info = self.code_generator.info
        self.manager = MidapManager()
        self.manager._setup_simulation(config, data_info, dram_info, dram_dict, processing_order)
        self.manager.setup_frame(0)
        mm = self.manager
        dram_info[MemoryType.Shared.value].tofile(os.path.join(out_dir_name, 'dram_shared.dat'))
        dram_info[MemoryType.Constant.value].tofile(os.path.join(out_dir_name, 'dram_wb.dat'))
        dram_info[MemoryType.Temporal.value].tofile(os.path.join(out_dir_name, 'dram_buf.dat'))
        dram_info[MemoryType.Output.value].tofile(os.path.join(out_dir_name, 'dram_output.dat'))
        for i in range(dram_info[MemoryType.Input.value].shape[0]):
            dram_info[MemoryType.Input.value][i].tofile(os.path.join(out_dir_name, 'dram_input_{}.dat'.format(i)))
        if generate_gold_data:
            dram_info[MemoryType.Shared.value].astype(np.uint8).tofile(os.path.join(out_dir_name, 'dram_shared.txt'), " ", "%02x")
            dram_info[MemoryType.Constant.value].astype(np.uint8).tofile(os.path.join(out_dir_name, 'dram_wb.txt'), " ", "%02x")
            dram_info[MemoryType.Temporal.value].astype(np.uint8).tofile(os.path.join(out_dir_name, 'dram_buf.txt'), " ", "%02x")
            dram_info[MemoryType.Output.value].astype(np.uint8).tofile(os.path.join(out_dir_name, 'dram_output.txt'), " ", "%02x")
            for i in range(dram_info[MemoryType.Input.value].shape[0]):
                dram_info[MemoryType.Input.value][i].astype(np.uint8).tofile(os.path.join(out_dir_name, 'dram_input_{}.txt'.format(i)), " ", "%02x")
        # Input is not required
        if integrate:
            with open(os.path.join(out_dir_name, out_prefix + postfix), 'wb') as fd_inst_bin:
                cg = self.code_generator
                load_info = []
                # fd_load_info.write("{}\n".format(len(cg.info[1])))
                inst_limit = 1024 # TODO: insert this value in the configuration
                transfer_info_idx_1 = 0
                transfer_info_idx_2 = 0
                for idx, layer_info in enumerate(cg.info[1]):
                    idx -= skipped_layer
                    layer_code = cg.generate_layer_code(layer_info)
                    instruction_left = len(layer_code) # TODO: considering synchronization...
                    offset = 0
                    while instruction_left > 0:
                        load_num = min(instruction_left, inst_limit)
                        instruction_left -= load_num
                        last = 1 if instruction_left == 0 else 0
                        load_info.append((last, addr + offset, load_num))
                        offset += load_num * 8
                    # fd_load_info.write("{} {}\n".format(addr, len(layer_code))) # to be fixed
                    if generate_gold_data:
                        mm.process_layer(layer_info)
                    if isinstance(layer_info.modules[0].op, HostProcessWrapper):
                        skipped_layer += 1
                        continue
                    gold_out_dir = None
                    if generate_gold_data:
                        gold_out_dir = os.path.join(out_dir_name, str(idx))
                        os.makedirs(gold_out_dir, exist_ok=True)
                        txt_file_name = os.path.join(gold_out_dir, 'inst.txt')
                        with open(txt_file_name, 'w') as fd_str:
                            for opcode in layer_code:
                                fd_str.write(str(opcode) + '\n')                        
                    for opcode in layer_code:
                        fd_inst_bin.write(opcode.to_binary())
                    require_pad = 0 if len(layer_code) % 8 == 0 else 8-(len(layer_code)%8)
                    for _ in range(require_pad):
                        fd_inst_bin.write(zero_pad)
                    size = 8 * (len(layer_code) + require_pad)
                    addr += size                        
                    if generate_gold_data:
                        mm.memory_controller.dump_gold_data(gold_out_dir)
                        request_array = cg.memory_op_generator.memory_op_module.get_request_array()
                        if config.DRAM.COMM_TYPE != 'TEST_3D':
                            np.savetxt(os.path.join(gold_out_dir, 'transfer_info.txt'), request_array[transfer_info_idx_1:].reshape(-1, 8).astype(np.uint32), fmt="%08x")
                            transfer_info_idx = len(request_array)
                        else:
                            np.savetxt(os.path.join(gold_out_dir, 'read_transfer_info.txt'), request_array[0][transfer_info_idx_1:].reshape(-1, RDMA3DDescriptor.num_of_fields).astype(np.uint32), fmt="%08x")
                            np.savetxt(os.path.join(gold_out_dir, 'write_transfer_info.txt'), request_array[1][transfer_info_idx_2:].reshape(-1, WDMA3DDescriptor.num_of_fields).astype(np.uint32), fmt="%08x")
                            transfer_info_idx_1 = len(request_array[0])
                            transfer_info_idx_2 = len(request_array[1])
                mom = cg.memory_op_generator.memory_op_module
                offset_info = rule.get_sram_offset_info(cg.info[0])
                load_info_array = []
                with open(os.path.join(out_dir_name, "load_info.txt"), 'w') as fd_load_info:
                    fd_load_info.write("{}\n".format(len(load_info)))
                    load_info_array.append(len(load_info))
                    mem_id = 0
                    transfer_type = rule.TransferType.Instruction
                    for first, src_addr, inst_num in load_info:
                        import math
                        transfer_size = math.ceil(inst_num / 8) * 64
                        if config.DRAM.COMM_TYPE != 'TEST_3D':
                            transfer_info = TransferInfo(
                                transfer_type,
                                mem_id + offset_info.imem_id_offset,
                                src_addr + (0 if cg.dynamic_base_addr else cfg.DRAM.OFFSET.INSTRUCTION),
                                offset_info.imem_pivot_address + offset_info.imem_addr_offset * mem_id,
                                transfer_size,
                                1,
                                0,
                                0)
                            id = len(mom.request_array)
                            fd_load_info.write("{} {} {}\n".format(first, id, mem_id * inst_limit + inst_num - 1))
                            load_info_array += [first, id, mem_id * inst_limit + inst_num -1]
                            mom.request_array.append(transfer_info)
                        else:
                            transfer_info = RDMA3DDescriptor(
                                acnt=transfer_size,
                                boff=transfer_size,
                                bcnt=1,
                                last=True,
                                baddr_ddr=src_addr + (0 if cg.dynamic_base_addr else cfg.DRAM.OFFSET.INSTRUCTION),
                                baddr_sram=offset_info.imem_pivot_address + offset_info.imem_addr_offset * mem_id,
                                transfer_type=transfer_type
                            )
                            id = len(mom.rdma_request_array)
                            fd_load_info.write("{} {} {}\n".format(first, id, mem_id * inst_limit + inst_num - 1))
                            load_info_array += [first, id, mem_id * inst_limit + inst_num -1]
                            mom.rdma_request_array.append(transfer_info)
                        mem_id = (mem_id + 1) % 2
                request_array = mom.get_request_array()
                if config.DRAM.COMM_TYPE != 'TEST_3D':
                    request_array.tofile(os.path.join(out_dir_name, 'transfer_info.dat'))
                    np.savetxt(os.path.join(out_dir_name, 'transfer_info.txt'), request_array.reshape(-1, 8).astype(np.uint32), fmt="%08x")
                else:
                    request_array[0].tofile(os.path.join(out_dir_name, 'read_transfer_info.dat'))
                    np.savetxt(os.path.join(out_dir_name, 'read_transfer_info.txt'), request_array[0].reshape(-1, RDMA3DDescriptor.num_of_fields).astype(np.uint32), fmt="%08x")
                    request_array[1].tofile(os.path.join(out_dir_name, 'write_transfer_info.dat'))
                    np.savetxt(os.path.join(out_dir_name, 'write_transfer_info.txt'), request_array[1].reshape(-1, WDMA3DDescriptor.num_of_fields).astype(np.uint32), fmt="%08x")
                load_info_array = np.array(load_info_array, dtype = np.uint32)
                load_info_array.tofile(os.path.join(out_dir_name, "load_info.dat"))
        else:
            cg = self.code_generator
            for idx, layer_info in enumerate(cg.info[1]):
                idx -= skipped_layer
                layer_code = cg.generate_layer_code(layer_info)
                output_file_name = os.path.join(out_dir_name, out_prefix + '{}_{}{}'.format(layer_info.name, idx, postfix))
                gold_out_dir = None
                if generate_gold_data:
                    mm.process_layer(layer_info)
                if isinstance(layer_info.modules[0].op, HostProcessWrapper):
                    skipped_layer += 1
                    continue
                if generate_gold_data:
                    gold_out_dir = os.path.join(out_dir_name, str(idx))
                    os.makedirs(gold_out_dir, exist_ok=True)
                    output_file_name = os.path.join(gold_out_dir, 'inst'+postfix)
                    txt_file_name = os.path.join(gold_out_dir, 'inst.txt')
                    gold_txt_file_name = os.path.join(gold_out_dir, 'inst_gold.txt')
                    gold_arr = b""
                    with open(txt_file_name, 'w') as fd_str:
                        for opcode in layer_code:
                            fd_str.write(str(opcode) + '\n')
                            gold_arr += bytes(opcode.to_binary())
                    np.frombuffer(gold_arr, dtype=np.uint8).tofile(gold_txt_file_name, " ", "%02x")
                    mm.memory_controller.dump_gold_data(gold_out_dir)
                with open(output_file_name, 'wb') as fd:
                    for opcode in layer_code:
                        fd.write(opcode.to_binary())

        if generate_gold_data and config.MODEL.NUM_FRAMES > 1:
            for frame_num in range(1, config.MODEL.NUM_FRAMES):
                skipped_layer = 0
                mm.setup_frame(frame_num)
                for idx, layer_info in enumerate(self.code_generator.info[1]):
                    idx -= skipped_layer
                    gold_out_dir = os.path.join(out_dir_name, str(idx))
                    mm.process_layer(layer_info)
                    if isinstance(layer_info.modules[0].op, HostProcessWrapper):
                        skipped_layer += 1
                        continue
                    mm.memory_controller.dump_gold_data(gold_out_dir, output_only=True)


    def _verify(
            self,
            out_dir_name = './microcode',
            out_prefix = 'default',
            debug_layers = [],
            *args,
            **kwargs):
        config, processing_order, dram_dict, dram_info, data_info = self.code_generator.info
        self.manager = MidapManager()
        self.manager._setup_simulation(config, data_info, dram_info, dram_dict, processing_order)
        self.manager.setup_frame(0)
        self.code_simulator = OpcodeSimulator()
        self.code_simulator.simulation_setup(config, self.manager)
        mm = self.manager
        cg = self.code_generator
        cs = self.code_simulator
        os.makedirs(out_dir_name, exist_ok=True)
        run_all = len(debug_layers) == 0
        for idx, layer_info in enumerate(processing_order):
            debug = layer_info.name in debug_layers
            layer_code = cg.generate_layer_code(layer_info)
            with open(os.path.join(out_dir_name, out_prefix + '_{}({}).txt'.format(layer_info.name, idx)), 'w') as fd_str:
                for opcode in layer_code:
                    fd_str.write(str(opcode) + '\n')
            # verification: compare original simulator results with the generated code simulator
            # for debug
            mm.pipeline[0].debug = debug
            cs.debug = debug
            #
            #if debug or idx == 0:
            cs.setup_status(cg.memory_op_generator.memory_op_module.request_array)
            mm.process_layer(layer_info)
            cs.run_and_compare(layer_code, debug or run_all)
            self.logger.info("Verification for layer {} is finished.".format(layer_info.name))

