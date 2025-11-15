import os
import sys

sys.path.append(os.path.dirname('./'))
sys.setrecursionlimit(10000)

from copy import copy
import models.transformer.gemma as gemma
from models.model_builder import ModelBuilder
from tools.test_system import parse
from config import cfg
from system_test_wrapper import SystemTestWrapper
from software.system_compiler.system_info import SystemInfo
from software.system_compiler.core_info import CoreInfo
from software.compiler.compile_info import CompileInfo


def parameter_setup(args):
    cfg.MIDAP.SYSTEM_WIDTH = args.system_width
    cfg.MIDAP.FMEM.NUM = args.fmem_config[0]
    cfg.MIDAP.FMEM.NUM_ENTRIES = args.fmem_config[1] * 1024
    cfg.MIDAP.WMEM.NUM = args.wmem_config[0]
    cfg.MIDAP.WMEM.NUM_ENTRIES = args.wmem_config[1] * 1024
    cfg.MIDAP.WMEM.E_NUM_ENTRIES = args.wmem_config[2] * 1024
    cfg.MIDAP.CONTROL_STRATEGY.WEIGHTED_SUM = not args.disable_weighted_sum
    ## TODO: System Compiler options
    cfg.MIDAP.CONTROL_STRATEGY.GRAPH_TRANSFORMER = args.graph_transformer
    cfg.MIDAP.CONTROL_STRATEGY.MAPPING_POLICY = args.mapping_policy
    cfg.MIDAP.CONTROL_STRATEGY.LOCAL_MEMORY_COMPILER = args.local_memory_compiler
    cfg.MIDAP.CONTROL_STRATEGY.L2_SPM_COMPILER = args.l2_spm_compiler 
    ## TODO : Core Compiler options
    ## TODO: Core Compiler
    cfg.MIDAP.CONTROL_STRATEGY.FIRST_LAYER = args.first_layer
    cfg.MODEL.ALLOW_ABSTRACT_DATA = not args.disable_abstract_layer
    cfg.MODEL.REDUCTION_LOGIC = not args.disable_reduction_layer
    cfg.MIDAP.WMEM.USE_EXTENSION = not args.disable_cim_extension
    cfg.MODEL.NUM_FRAMES = args.num_frames

    ## Simulation configurations
    cfg.SYSTEM.FREQUENCY = args.system_freq
    cfg.DRAM.CLEAR_FRAME_MEM = args.clear_frame
    cfg.DRAM.BUS_BANDWIDTH = args.bus_bandwidth
    cfg.DRAM.FREQUENCY = args.dram_freq
    cfg.DRAM.NUM_CHANNELS = args.dram_chan
    cfg.DRAM.PAGE_SIZE = 4096 * args.dram_chan
    cfg.DRAM.COMM_TYPE = args.dram_comm_type
    cfg.DRAM.OFFSET.SHARED = args.mem_offset[0]
    cfg.DRAM.OFFSET.INPUT = args.mem_offset[1] + 0x80000000
    cfg.DRAM.OFFSET.OUTPUT = args.mem_offset[2] + 0x80000000
    cfg.DRAM.OFFSET.WEIGHT_BIAS = args.mem_offset[3]
    cfg.DRAM.OFFSET.BUFFER = args.mem_offset[4] + 0x100000000   # Give more flexibility to the temp. section
    cfg.SYSTEM.BANDWIDTH = (
        cfg.DRAM.CHANNEL_SIZE * cfg.DRAM.FREQUENCY * cfg.DRAM.NUM_CHANNELS * 2
    ) / cfg.SYSTEM.DATA_SIZE

    cfg.MIDAP.FUNCTIONAL_SIM = args.functional_simulation
    if args.quantize == False:
        cfg.MIDAP.PACKET_SIZE = int(args.packet_size // 4)
    else:
        cfg.MIDAP.PACKET_SIZE = int(args.packet_size)

    # Debug information
    if args.debug:
        cfg.LOGGING_CONFIG_DICT["root"]["level"] = "DEBUG"
        cfg.LOGGING_CONFIG_DICT["root"]["handlers"] = ["console", "file"]
        cfg.LOGGING_CONFIG_DICT["loggers"]["debug"]["level"] = "DEBUG"


def merge_core_info(info_list) -> CoreInfo:
    merged_model = copy(info_list[0].compile_info.model)
    for info in info_list[1:]:
        for layer in info.compile_info.model.layers:
            merged_model.register_layer(layer)
    result = CoreInfo(core_id=1, compile_info=CompileInfo(merged_model))
    result.local_memory_info.required_size = max((info.local_memory_info.required_size for info in info_list))
    for info in info_list:
        result.compile_info.layer_dict.update(info.compile_info.layer_dict)
        result.compile_info.wmem_info_dict.update(filter(lambda x: x[0] in info.local_memory_info.write_mem_pivot_dict, info.compile_info.wmem_info_dict.items()))
        result.local_memory_info.address_dict.update(info.local_memory_info.address_dict)
        result.local_memory_info.write_mem_pivot_dict.update(info.local_memory_info.write_mem_pivot_dict)
    return result


if __name__ == '__main__':
    args = parse()
    parameter_setup(args)
    cfg.MODEL.QUANTIZED = args.quantize

    # # Gemma2-2B specific parameters
    input_len = args.input_size if args.input_size > 0 else 1
    input_pos = 0
    hidden_size = 2304
    intermediate_size = 9216
    num_attention_heads = 8
    num_key_value_heads = 4
    head_dim = 256
    num_hidden_layers = 26

    if args.network == 'gemma2-2b':
        mb = gemma.Gemma2_2b(input_len=input_len, input_pos=input_pos, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim, num_hidden_layers=num_hidden_layers)
    elif args.network == 'gemma2_layer_pim_offload':
        mb = gemma.GemmaLayer_pim_offload(input_len=input_len, input_pos=input_pos, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim)
    elif args.network == 'gemma2-2b_pim_offload':
        mb = gemma.Gemma2_2b_pim_offload(input_len=input_len, input_pos=input_pos, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim, num_hidden_layers=num_hidden_layers)
    sys_wrapper = SystemTestWrapper(core_idx=1, num_cores=num_hidden_layers)
    sys_wrapper.setup_from_builder(mb)
    sys_wrapper.compile(quantize=args.quantize, spm_num_banks=0, spm_bank_size=0)
    sys_wrapper.core_info = [merge_core_info(sys_wrapper.core_info)]
    sys_wrapper.cm.core_info = sys_wrapper.core_info
    sys_wrapper.cm.memory_info.local_memory_info_dict.clear()
    sys_wrapper.cm.memory_info.local_memory_info_dict[sys_wrapper.core_idx] = sys_wrapper.core_info[0].local_memory_info
    kwargs = {'prefix': 'Gemma2-2b', 'quantize': args.quantize}
    if args.save_dir:
        kwargs['save_dir'] = args.save_dir
    sys_wrapper.save_compiled_info(**kwargs)
