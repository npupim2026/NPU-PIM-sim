from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy as np

from midap_backend.wrapper.op_wrapper import ConvPoolWrapper, ConvWrapper, DWWrapper, PoolWrapper, DWConvWrapper, ArithmeticWrapper
from midap_simulator.dataflow import generate_dataflow_info

from .level0 import ControlLogicLv1

class ControlLogicLv2(ControlLogicLv1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_generator(self, *args, **kwargs):
        super().set_generator(*args, **kwargs)
        self.read_crit_idx = -1

    def add_write_dram(self, access_count=1):
        self.manager.stats.write_dram(self.concurrency * access_count)
    
    def convpool_generator(self, head_x, tail_x, filter_idx, z_iter, *args, **kwargs):
        main_op = self.main_op
        s = main_op.stride
        yz_worker_flag = self.input_tensor.mapping_type == "default"
        for x in range(head_x, tail_x + 1, s):
            first = True
            for y in range(self.head_y, self.tail_y + 1, s):
                for filter_offset in range(self.z_iter):
                    if first:
                        worker = self.convpool_worker(x, y, filter_idx, filter_offset)
                        for dataflow in worker:
                            yield dataflow
                        first = False
                    else:
                        self.update_stats(x, y)
        self.generator = self.default_generator()

    def update_stats(self, x, y, include_last=True):
        main_op = self.main_op
        access_count = 0
        wmem_factor = 1
        if isinstance(main_op, ArithmeticWrapper):
            raise ValueError(
                "main operation {} must not call this function".format(main_op)
            )
        elif (any([
            isinstance(main_op, ConvWrapper) and self.input_tensor.mapping_type != "default",
            isinstance(main_op, DWWrapper)]
            )
        ):
            it = self.input_tensor
            wmem_factor = 0
            read_factor = 1
            if isinstance(main_op, ConvWrapper):
                read_factor = it.shape[-1] // self.system_width
                wmem_factor = 1
            if isinstance(main_op, DWConvWrapper):
                read_factor = 1
                wmem_factor = 1
            min_x, max_x = max(x, 0), min(x + main_op.k_w - 1, it.shape[0] - 1)
            min_y, max_y = max(y, 0), min(y + main_op.k_h - 1, it.shape[1] - 1)
            if self.input_tensor.mapping_type == "valid":
                it = self.input_tensor
                scale_x, scale_y, _ = it.scale
                x_access_count = max_x // scale_x - (min_x - 1) // scale_x
                y_access_count = max_y // scale_y - (min_y - 1) // scale_y
            else:
                x_access_count = max_x - min_x + 1
                y_access_count = max_y - min_y + 1
            access_count = x_access_count * y_access_count * read_factor
        elif all(
            [isinstance(main_op, ConvWrapper), self.input_tensor.mapping_type == "default"]
        ):
            main_op = self.main_op
            k_h, k_w = main_op.k_h, main_op.k_w
            in_w, in_h, in_c = self.input_tensor.orig_shape
            row_per_kernel_yz = math.ceil(in_c * k_h/self.system_width)
            yz_plane_size = in_h * in_c
            in_x, in_y = x, y
            for kx in range(k_w):  # ZY-plane wise multiplication w/ alignment submodule
                x = in_x + kx
                if x < 0 or x >= in_w:
                    continue
                last_x = (x == in_w - 1) or (kx == k_w - 1)
                fmem_idx, effective_x = self.get_fmem_info(x)
                # wmem configuration
                start_ky = max(0, -in_y)
                end_ky = min(k_h, in_h - in_y)
                wmem_offset = (start_ky * in_c) % self.system_width
                fmem_start_address = (
                    effective_x * yz_plane_size + (in_y + start_ky) * in_c
                )
                fmem_offset = fmem_start_address % self.system_width
                bubble = wmem_offset - fmem_offset
                access_count += math.ceil(end_ky * in_c/self.system_width) - (
                    start_ky * in_c // self.system_width
                )
                if bubble < 0:
                    access_count += 1
        else:
            raise ValueError("Not Classified main operation {}".format(main_op))
        # self.logger.debug("Computation time for ({},{}) is estimated as {} cycles".format(x, y, access_count))
        if not include_last:
            access_count -= 1
        self.manager.stats.read_fmem(access_count)
        self.manager.stats.read_wmem(access_count * wmem_factor)
        self.manager.stats.increase_cycle(access_count)
        # self.skipped_cycles += access_count

    def depthwise_worker(self, in_x, in_y, filter_idx, filter_offset, *args, **kwargs):
        main_op = self.main_op
        orig_x, _, _ = self.input_tensor.get_loc(
            (min(self.input_tensor.shape[0], in_x + main_op.k_w) - 1, 0, 0)
        )
        fmem_idx, _ = self.get_fmem_info(orig_x)
        yield generate_dataflow_info(
            phase=1,
            loc=self.output_loc,
            fmem_idx=fmem_idx,
            fmem_row=0,
            wmem_row=0 if isinstance(self.main_op, ConvWrapper) else -1,
            reset=True,
            last=True,
        )
        self.update_stats(in_x, in_y, False)

    def conv_z_worker(self, in_x, in_y, filter_idx, filter_offset, *args, **kwargs):
        main_op = self.main_op
        orig_x, _, _ = self.input_tensor.get_loc(
            (min(self.input_tensor.shape[0], in_x + main_op.k_w) - 1, 0, 0)
        )
        fmem_idx, _ = self.get_fmem_info(orig_x)
        yield generate_dataflow_info(
            phase=1,
            loc=self.output_loc,
            fmem_idx=fmem_idx,
            fmem_row=0,
            wmem_row=0,
            reset=True,
            last=True,
        )
        self.update_stats(in_x, in_y, False)

    def arithmetic_worker(self, x, z_pivot, z_iter, *args, **kwargs):
        main_op = self.main_op
        _, in_h, in_c = self.shape
        in_x, _, _ = self.input_tensor.get_loc((x, 0, 0))
        fmem_idx, effective_x = self.get_fmem_info(in_x)
        row_per_channel = in_c // self.system_width
        access_count = row_per_channel * in_h - 1
        yield generate_dataflow_info(
            phase=1,
            loc=(x, 0, z_pivot),
            fmem_idx=fmem_idx,
            fmem_row=0,
            wmem_row=0,
            reset=True,
            last=True,
        )
        self.manager.stats.increase_cycle(access_count)
        # self.skipped_cycles += access_count
        self.manager.stats.read_fmem(access_count)
        self.manager.stats.read_wmem(access_count)


# class ControlLogicLv3(ControlLogicLv2):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # self.skip_write = True
#         self.first_write_time = -1
#         self.dram_write_count = 0
#         self.set_next_info = None
#         self.restored = False

#     def setup(self, *args, **kwargs):
#         super().setup(*args, **kwargs)
#         self.restored = False
#         self.set_next_info = None
#         self.first_write_time = -1
#         self.dram_write_count = 0

#     def set_generator(self, *args, **kwargs):
#         self.restored = False
#         super().set_generator(*args, **kwargs)

#     def set_next(self, last):
#         self.set_next_info = (last, self.skipped_cycles)
#         return super().set_next(last)

#     def set_next_real(self, last):
#         return super().set_next(last)

#     def generate(self, dataflow, last_filter=False):
#         running_info = RunningInfo(x=self.output_loc[0], last_filter=last_filter)
#         simulated_cycle = 1 + self.skipped_cycles
#         self.skipped_cycles = 0
#         self.loaded_fmem = (dataflow.fmem_idx, dataflow.fmem_row)
#         set_next_info = self.set_next_info
#         self.set_next_info = None
#         return (dataflow, [set_next_info, running_info, simulated_cycle])

#     def add_write_dram(self, access_count=1):
#         super().add_write_dram(access_count)
#         if self.first_write_time == -1:
#             self.first_write_time = (
#                 self.manager.stats.total_cycle() + self.skipped_cycles
#             )
#         self.dram_write_count += access_count

#     def sync(self):
#         super().sync()
#         if self.dram_write_count > 0:
#             # self.logger.info("Time: {}, DRAM Write Update info: {}, {}".format(
#             #    self.manager.stats.current_cycle(), self.first_write_time, self.dram_write_count))
#             self.manager.memory_controller.memory_manager.write_dram_bulk(
#                 self.concurrency, self.dram_write_count, self.first_write_time
#             )
#             self.first_write_time = -1
#             self.dram_write_count = 0
