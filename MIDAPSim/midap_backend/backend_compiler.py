import logging

from numpy.core.fromnumeric import shape
from midap_backend.wrapper.compile_wrapper import CompileWrapper
from software.system_compiler.memory_info import MemoryType
from midap_backend.wrapper.layer_wrapper import LayerWrapper
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import numpy.typing as npt
import copy
import yaml
import math

# For Quantization
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from quant_op.mq import QuantOps
##

from collections import OrderedDict

from config import cfg
from software.network.types import ActivationType

from .wrapper.info import QuantInfo, WriteInfo
from .wrapper.op_wrapper import ConvPoolWrapper, ConvWrapper, DWWrapper, DWConvWrapper, PoolWrapper, ArithmeticWrapper, AddWrapper, SumWrapper, MulWrapper, AvgpoolWrapper, MaxpoolWrapper, UpBilinearWrapper, MatMulWrapper, RoPEWrapper

from logger import init_logger

class BackendCompiler:
    def __init__(self, compile_info: CompileWrapper):
        self.compile_info = compile_info
        self.addr_dict = {}
        self.dram_data = [np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)]
        self.sync_info = {}
        # logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        # self.logger = logging.getLogger("debug")
        self.logger = init_logger("Backend Compiler", logging.INFO)

    def setup_memory(
        self,
        mem_info : Dict[MemoryType, Dict[str, Any]]
        ):
        for mem_id, info in mem_info.items():
            self.dram_data[mem_id.value] = info["data"]
            if mem_id == MemoryType.Shared:
                self.compile_info.config.DRAM.OFFSET.SHARED = info["offset"]
            elif mem_id == MemoryType.Constant:
                self.compile_info.config.DRAM.OFFSET.WEIGHT_BIAS = info["offset"]
            elif mem_id in [MemoryType.Temporal, MemoryType.SPM]:
                self.compile_info.config.DRAM.OFFSET.BUFFER = info["offset"]
            elif mem_id == MemoryType.Input:
                self.compile_info.config.DRAM.OFFSET.INPUT = info["offset"]
            elif mem_id == MemoryType.Output:
                self.compile_info.config.DRAM.OFFSET.OUTPUT = info["offset"]

    def compile(
        self,
        shared_addr_dict = None,
        **kwargs):
        #TODO Relocate all backend compile methods to software.compiler or software.system_compiler
        # 4. Determine Write Logic handling
        self.determine_write_strategy()
        # 5. Add Sync Operation (for multi-midap)
        # self.add_sync_info(sync_layer_dict, sync_information)
        # 7. DRAM Allocation?
        self.dram_allocation(shared_addr_dict)

    # 2. Update WMEM Prepare Information
    # 3. Update write_on_dram_pivot 

    # Determine Write logic type & z_tiling
    def determine_write_strategy(self):
        tensor_dict = self.compile_info.tensor_dict
        config = self.compile_info.config
        layers = self.compile_info.layers

        for key in layers:
            layer = layers[key]
            pi = layer.processing_info
            ot = layer.main_output
            op = layer.main_op
            filter_size = 0
            num_filter_per_wmem = 1
            compute_unit = 1
            # self.logger.info(op)
            # self.logger.info(ot.shape)
            if isinstance(op, ConvWrapper):
                compute_unit = cfg.MIDAP.WMEM.NUM
                weight = tensor_dict[0][op.weight].tensor
                filter_size = weight[0,:].size
                num_filter_per_wmem = weight.shape[0] // compute_unit
                z_tile_size = num_filter_per_wmem
            elif isinstance(op, RoPEWrapper):
                compute_unit = cfg.MIDAP.SYSTEM_WIDTH
                weight = tensor_dict[0][op.weight].tensor
                filter_size = weight[0, :].size
                num_filter_per_wmem = weight.shape[0]
                z_tile_size = num_filter_per_wmem
            else:
                compute_unit = cfg.MIDAP.SYSTEM_WIDTH
                z_tile_size = ot.shape[2] // compute_unit
                if isinstance(op, UpBilinearWrapper):
                    filter_size = tensor_dict[0][op.weight].tensor.size
                elif isinstance(op, DWConvWrapper) or isinstance(op, MatMulWrapper):
                    filter_size = tensor_dict[0][op.weight].tensor.size // z_tile_size
                    assert filter_size == tensor_dict[0][op.weight].tensor[0].size or isinstance(op, MatMulWrapper)
                    num_filter_per_wmem = z_tile_size 
                elif isinstance(op, ArithmeticWrapper):
                    f = op.in2
                    filter_size = f.orig_shape[1] * f.orig_shape[2]
            stride = 1
            x_pivot = 0
            unit_size = 0
            y_size = ot.shape[1]
            if isinstance(op, ConvPoolWrapper):
                stride = op.stride
                x_pivot = op.pad_l
            # Type 1: Save Whole ZY Plane 
            if ot.shape[0] <= pi.write_on_dram_pivot:
                if isinstance(op, ConvWrapper) and filter_size * num_filter_per_wmem <= config.MIDAP.WMEM.NUM_ENTRIES:
                    pi.behavior_info.in_z_tile = z_tile_size
                elif all([isinstance(op, DWWrapper) or isinstance(op, ArithmeticWrapper) or isinstance(op, SumWrapper), filter_size * num_filter_per_wmem <= config.MIDAP.WMEM.E_NUM_ENTRIES ]):
                    pi.behavior_info.in_z_tile = z_tile_size
                # TODO: Find optimized tile size, group_size
                continue # This layer does not have off-chip write
            if all([
                ot.shape[2] == ot.orig_shape[2],
                ot.orig_shape[2] <= config.MIDAP.WRITE_BUFFER.NUM_ENTRIES,
                (isinstance(op, ConvWrapper) and filter_size * num_filter_per_wmem <= config.MIDAP.WMEM.NUM_ENTRIES
                 and (num_filter_per_wmem * compute_unit == ot.orig_shape[2] or config.DRAM.COMM_TYPE != 'TEST_3D')) or
                ((isinstance(op, DWWrapper) or isinstance(op, ArithmeticWrapper) or isinstance(op, SumWrapper)) and filter_size * num_filter_per_wmem <= config.MIDAP.WMEM.E_NUM_ENTRIES),
                layer.input_tensor.mapping_type != 'valid' or config.DRAM.COMM_TYPE != 'TEST_3D'
            ]):
                pi.write_logic_type = 1
                for i in range(1, ot.shape[1] + 1):
                    y_size = math.ceil(ot.shape[1] / i)
                    if y_size * ot.orig_shape[2] <= config.MIDAP.WRITE_BUFFER.NUM_ENTRIES:
                        break
                pi.behavior_info.in_z_tile = z_tile_size
                unit_size = y_size * ot.orig_shape[2]
            # elif ot.shape[1] != ot.orig_shape[1]:
            #     if isinstance(op, ConvWrapper):
            #         z_tile_size = config.MIDAP.SYSTEM_WIDTH // compute_unit
            #         if z_tile_size * filter_size > config.MIDAP.WMEM.NUM_ENTRIES:
            #             raise RuntimeError("[Virtualized Tensor: Type 2] Minimum size requirement for write buffer is not satisfied with layer {}, tensor {}, op {}".format(layer, ot, op))
            #     else:
            #         z_tile_size = 1
            #     group_size = z_tile_size
            #     pi.behavior_info.in_z_tile = z_tile_size
            else:
                if isinstance(op, ConvWrapper):
                    if (ot.orig_shape[2] <= config.MIDAP.WRITE_BUFFER.NUM_ENTRIES and filter_size * num_filter_per_wmem <= config.MIDAP.WMEM.NUM_ENTRIES):
                        z_size = ot.shape[2]
                    else:
                        z_size = config.MIDAP.SYSTEM_WIDTH
                        while ot.shape[2] % z_size != 0:
                            z_size = z_size // 2
                    if z_size < compute_unit:
                        raise RuntimeError
                    z_tile_size = z_size // compute_unit 
                    if ot.orig_shape[2] % compute_unit != 0:
                        raise RuntimeError("Write Logic cannot be handled for layer {}, otensor {}, op: {} for unz_tile_sizeed dram output issue".format(layer, ot, op))
                    # if z_tile_size * filter_size <= config.MIDAP.WMEM.NUM_ENTRIES:
                    if pi.wmem_strategy.group_size >= z_tile_size:
                        #if num_filter_per_wmem * filter_size <= config.MIDAP.WMEM.NUM_ENTRIES:
                        #    group_size = num_filter_per_wmem 
                        #else:
                        pi.behavior_info.in_z_tile = z_tile_size
                else:
                    if isinstance(op, ArithmeticWrapper):
                        raise NotImplementedError
                    if not (ot.orig_shape[2] <= config.MIDAP.WRITE_BUFFER.NUM_ENTRIES and filter_size * num_filter_per_wmem <= config.MIDAP.WMEM.E_NUM_ENTRIES):
                        z_tile_size = 1
                    # while True:
                    #     next_z = z_tile_size * 2
                    #     if num_filter_per_wmem % next_z > 0 or pi.wmem_strategy.group_size % next_z > 0:
                    #         break
                    #     z_tile_size = next_z
                # If All Z'YX' plane can be located in write_buffer: Type 3 / else: Type 2
                for i in range(1, ot.shape[1] + 1):
                    y_size = math.ceil(ot.shape[1] / i)
                    if y_size * compute_unit * z_tile_size <= config.MIDAP.WRITE_BUFFER.NUM_ENTRIES:
                        break
                unit_size = y_size * compute_unit * z_tile_size
                if unit_size > config.MIDAP.WRITE_BUFFER.NUM_ENTRIES:
                    raise RuntimeError("Minimum size requirement for write buffer is not satisfied with layer {}, tensor {}, op {}".format(layer, ot, op))
                maximum_x_size = 1
                for action, idx, min_x, max_x in pi.behavior_info:
                    if action not in ['PROCESS']:
                        continue
                    min_ox = (min_x + x_pivot) // stride
                    max_ox = (max_x + x_pivot) // stride
                    if pi.reverse_write:
                        min_ox, max_ox = (ot.shape[0] - max_ox - 1, ot.shape[0] - min_ox - 1)
                    min_ox = max(pi.write_on_dram_pivot, min_ox)
                    x_len = max_ox - min_ox + 1
                    maximum_x_size = max(x_len, maximum_x_size)
                # Write Logic Type: 2
                # Type 3 is not allowed for 3D DMA
                if cfg.DRAM.COMM_TYPE == 'TEST_3D' or ot.mapping_type == 'wmem' or isinstance(op, ArithmeticWrapper) or maximum_x_size * unit_size > config.MIDAP.WRITE_BUFFER.NUM_ENTRIES:
                    pi.write_logic_type = 2
                    if z_tile_size > pi.wmem_strategy.group_size and not any([isinstance(op, PoolWrapper), isinstance(op, SumWrapper), isinstance(op, UpBilinearWrapper)]):
                        z_tile_size = pi.wmem_strategy.group_size
                    pi.behavior_info.in_z_tile = z_tile_size
                else: # Write Logic Type: 3
                    pi.write_logic_type = 3
                    pi.behavior_info.out_z_tile = z_tile_size // pi.behavior_info.in_z_tile
            # Setup Write Information per each 'PROCESS' Behavior
            for behavior in pi.behavior_info:
                if behavior.type not in ['PROCESS']:
                    continue
                _, _, min_x, max_x = behavior
                min_ox = (min_x + x_pivot) // stride
                max_ox = (max_x + x_pivot) // stride
                if pi.reverse_write:
                    min_ox, max_ox = (ot.shape[0] - max_ox - 1, ot.shape[0] - min_ox - 1)
                min_ox = max(pi.write_on_dram_pivot, min_ox)
                x_len = (max_ox - min_ox + 1)
                if x_len > 0:
                    write_x_size = x_len if pi.write_logic_type == 3 \
                        else max(1, cfg.MIDAP.WRITE_BUFFER.NUM_ENTRIES // (ot.shape[1] * ot.orig_shape[2]))
                    if ot.mapping_type == 'wmem':
                        assert pi.write_logic_type != 3
                        write_x_size = 1
                    write_z_size = ot.orig_shape[-1] if pi.write_logic_type == 1 else z_tile_size * compute_unit
                    write_shape = (write_x_size, y_size, write_z_size)
                    write_unit = write_z_size if pi.write_logic_type > 1 else np.prod(write_shape)
                    # if write_unit % 32 != 0:
                    #     raise RuntimeError(f"Cannot satisfy DMA Requirements: {ot}")
                    write_info = WriteInfo(pi.write_logic_type, write_unit, write_shape, min_ox)
                    behavior.write_info = write_info
    # # Add Sync op
    # def add_sync_info(self, layer_id_dict : Dict[str, Any], sync_information: Dict[str, List[Any]]):
    #     if sync_information is None:
    #         return
    #     layers = self.compile_info.layers
    #     for key, layer in layers.items():
    #         # All layers must have own sync id
    #         sync_id = layer_id_dict[key]
    #         wait_list = []
    #         if key in sync_information:
    #             wait_list = sync_information[key]
    #         # for b in layer.processing_info.behavior_info:
    #         #     tp, in1, in2, in3 = b
    #         #     if tp == 'LOAD':
    #         #         if in2 in layer_id_dict:
    #         #             wait_list.append(layer_id_dict[in2])
    #         #         break
    #         layer.sync_info = SyncInfo(sync_id, wait_list)
    #         self.logger.debug(f"Update sync info for layer {layer}: id = {sync_id}, wait_list = {wait_list}")
    
    def dram_allocation(self, shared_addr_dict = None):
        config = self.compile_info.config
        tensor_dict = self.compile_info.tensor_dict
        saddr_dict = {} if shared_addr_dict is None else shared_addr_dict # TODO      
        # for data_name in tensor_dict:
        #     if data_name in saddr_dict:
        #         self.addr_dict[data_name] = saddr_dict[data_name]
        #         continue
        self.addr_dict.update(saddr_dict)
