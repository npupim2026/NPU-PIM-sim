from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import math

from midap_backend.wrapper.op_wrapper import ConvPoolWrapper, ConvWrapper, DWWrapper, PoolWrapper, DWConvWrapper, ArithmeticWrapper, UpBilinearWrapper
from midap_simulator.dataflow import generate_dataflow_info

from .base import ControlLogic, ControlLogicBase

class ControlLogicLv0(ControlLogicBase):
    def __init__(self, manager):
        super().__init__(manager)

    def depthwise_worker(self, in_x, in_y, filter_idx, filter_offset):
        main_op = self.main_op
        load_weight = False
        dilation = 1
        if isinstance(main_op, DWConvWrapper):
            load_weight = True
            dilation = main_op.dilation
        k_h, k_w = main_op.k_h, main_op.k_w
        in_w, in_h, in_c = self.input_tensor.shape
        row = filter_idx // self.system_width + filter_offset
        wmem_row_offset = filter_offset * k_h * k_w
        _, real_h, real_c = self.input_tensor.orig_shape
        yz_plane_size = real_h * real_c
        reset = True
        dataflow_info = None
        for kx in range(k_w):
            x = in_x + kx * dilation
            orig_x = x
            if x < 0 or x >= in_w:
                if not main_op.in_plane:
                    continue
                x = min(max(x, 0), in_w-1)
            for ky in range(k_h):
                y = in_y + ky * dilation
                if not self.input_tensor.valid(orig_x, y):
                    if not self.config.MIDAP.EFFICENT_LOGIC:
                        self.skipped_cycles += 1
                    continue
                if y < 0 or y >= in_h:
                    if not main_op.in_plane:
                        continue
                    y = min(max(y, 0), in_h-1)
                mapped_x, mapped_y, _ = self.input_tensor.get_loc((x, y, 0))
                fmem_idx, effective_x = self.get_fmem_info(mapped_x)
                # Error Checking
                fmem_row = (
                    effective_x * yz_plane_size + mapped_y * real_c
                ) // self.system_width + row
                wmem_row = -1
                if load_weight:
                    # wmem_row = (
                    #     kx * k_h * real_c + ky * real_c
                    # ) // self.system_width + row
                    wmem_row = (
                        kx * k_h + ky + (wmem_row_offset if not isinstance(main_op, UpBilinearWrapper) else 0)
                    )
                if dataflow_info is not None:
                    yield dataflow_info
                dataflow_info = generate_dataflow_info(
                    phase=1,
                    loc=self.output_loc,
                    fmem_idx=fmem_idx,
                    fmem_row=fmem_row,
                    wmem_row=wmem_row,
                    reset=reset,
                )
                reset = False
        if dataflow_info is not None:
            dataflow_info.last = True
            yield dataflow_info

    def conv_z_worker(self, in_x, in_y, filter_idx, filter_offset, *args, **kwargs):
        main_op = self.main_op
        dilation = main_op.dilation
        k_h, k_w = main_op.k_h, main_op.k_w
        real_w, real_h, real_c = self.input_tensor.orig_shape
        row_per_channel = real_c // self.system_width
        yz_plane_rows = real_h * row_per_channel
        in_w, in_h, in_c = self.input_tensor.shape
        reset = True
        dataflow_info = None
        for kx in range(k_w):
            x = in_x + kx * dilation
            if x < 0 or x >= in_w:
                continue
            for ky in range(k_h):
                y = in_y + ky * dilation
                if y < 0 or y >= in_h:
                    continue
                if not self.input_tensor.valid(x, y):
                    if not self.config.MIDAP.EFFICENT_LOGIC:
                        self.manager.stats.increase_cycle()
                        if self.config.MIDAP.CORE_ID >= 0:
                            self.memory_controller.elapse_cycle()
                    continue
                mapped_x, mapped_y, _ = self.input_tensor.get_loc((x, y, 0))
                fmem_idx, effective_x = self.get_fmem_info(mapped_x)
                fmem_start_row = (
                    effective_x * yz_plane_rows + mapped_y * row_per_channel
                )
                wmem_start_row = self.num_filter_rows * filter_offset + kx * k_h * row_per_channel + ky * row_per_channel
                for row in range(row_per_channel):
                    if dataflow_info is not None:
                        yield dataflow_info
                    dataflow_info = generate_dataflow_info(
                        phase=1,
                        loc=self.output_loc,
                        fmem_idx=fmem_idx,
                        fmem_row=fmem_start_row + row,
                        wmem_row=wmem_start_row + row,
                        reset=reset,
                    )
                    reset = False
        if dataflow_info is not None:
            dataflow_info.last = True
        yield dataflow_info

    def conv_yz_worker(self, in_x, in_y, filter_idx, filter_offset, *args, **kwargs):  # default input tensor type
        main_op = self.main_op
        k_h, k_w = main_op.k_h, main_op.k_w
        in_w, in_h, in_c = self.input_tensor.orig_shape
        row_per_kernel_yz = math.ceil(in_c * k_h/self.system_width)
        yz_plane_size = in_h * in_c
        reset = True
        for kx in range(k_w):  # ZY-plane wise multiplication w/ alignment submodule
            x = in_x + kx
            if x < 0 or x >= in_w:
                continue
            last_x = (x == in_w - 1) or (kx == k_w - 1)
            fmem_idx, effective_x = self.get_fmem_info(x)
            # wmem configuration
            start_ky = max(0, -in_y)
            end_ky = min(k_h, in_h - in_y)
            wmem_start_row = filter_offset * self.num_filter_rows + kx * row_per_kernel_yz
            wmem_start_row += start_ky * in_c // self.system_width  # when pad skipped
            wmem_offset = (start_ky * in_c) % self.system_width
            fmem_start_address = effective_x * yz_plane_size + (in_y + start_ky) * in_c # FO
            fmem_offset = fmem_start_address % self.system_width
            fmem_start_row = fmem_start_address // self.system_width
            fmem_last_row = math.ceil((effective_x * yz_plane_size + (in_y + end_ky) * in_c)/self.system_width) - 1

            bubble = wmem_offset - fmem_offset
            num_rows = math.ceil(end_ky * in_c/self.system_width) - (
                start_ky * in_c // self.system_width
            )
            self.save_last_run = (
                fmem_start_row,
                wmem_start_row,
                num_rows,
                fmem_offset,
                wmem_offset,
                end_ky,
                k_h,
                row_per_kernel_yz,
                fmem_last_row,
                0,
            )
            for row in range(num_rows):
                if bubble < 0:
                    # if self.loaded_fmem != (fmem_idx, fmem_start_row):
                    yield generate_dataflow_info(
                        phase=1,
                        loc=self.output_loc,
                        fmem_idx=fmem_idx,
                        fmem_row=fmem_start_row,
                        reset=reset,
                        junk=True,
                    )
                    bubble = self.system_width + bubble
                    fmem_start_row += 1
                fmem_row = (
                    fmem_last_row
                    if fmem_last_row < fmem_start_row + row
                    else fmem_start_row + row
                )
                wmem_row = wmem_start_row + row
                cnt = 0
                last = False
                if row == num_rows - 1:
                    cnt = (end_ky * in_c) % self.system_width
                    last = last_x
                yield generate_dataflow_info(
                    phase=1,
                    loc=self.output_loc,
                    fmem_idx=fmem_idx,
                    fmem_row=fmem_row,
                    wmem_row=wmem_row,
                    broadcast_offset=bubble,
                    delete_foffset=wmem_offset,
                    delete_boffset=cnt,
                    reset=reset,
                    last=last,
                )
                wmem_offset = 0
                reset = False

    def matmul_worker(self, in_x, in_y, filter_idx, filter_offset):
        main_op = self.main_op
        fmem_idx, effective_x = self.get_fmem_info(in_x)
        fmem_row_base = self.input_tensor.get_address((effective_x, in_y, 0)) // self.system_width
        wmem_row_base = filter_offset * self.input_tensors[-1].shape[0]
        #assert self.input_tensor.shape[-1] % self.system_width == 0
        #row_iter = self.input_tensor.shape[-1] // self.system_width
        for i in range(self.input_tensors[-1].shape[0]):
            fmem_row = fmem_row_base + i // self.system_width
            fmem_col = i % self.system_width
            wmem_row = wmem_row_base + i
            yield generate_dataflow_info(
                phase=1,
                reset=(i==0),
                last=(i==self.input_tensors[-1].shape[0]-1),
                loc=self.output_loc,
                fmem_idx=fmem_idx,
                fmem_row=fmem_row,
                wmem_row=wmem_row,
                fmem_col_broadcast=fmem_col,
            )

    def arithmetic_worker(self, x, z_pivot, z_iter, *args, **kwargs):
        main_op = self.main_op
        _, in_h, in_c = self.shape
        in_x, _, _ = self.input_tensor.get_loc((x, 0, 0))
        fmem_idx, effective_x = self.get_fmem_info(in_x)
        row_per_channel = in_c // self.system_width
        for y in range(in_h):
            fmem_start_row = (
                self.input_tensor.get_address((effective_x, y, z_pivot)) // self.system_width
            )
            for row in range(z_pivot, z_pivot + z_iter):  # I'm not sure that input virtualization can be applied on arithmetic operators
                z = row * self.system_width
                wmem_row = row_per_channel * y + row if not main_op.broadcast else row
                self.output_loc = (x, y, z)
                yield generate_dataflow_info(
                    phase=1,
                    loc=self.output_loc,
                    reset=True,
                    last=True,
                    fmem_idx=fmem_idx,
                    fmem_row=fmem_start_row + row,
                    wmem_row=wmem_row,
                )

    def weighted_sum_worker(self, x, z_pivot, z_iter, *args, **kwargs):
        main_op = self.main_op
        _, in_h, in_c = self.shape
        in_x, _, _ = self.input_tensor.get_loc((x, 0, 0))
        fmem_idx, effective_x = self.get_fmem_info(in_x)
        fmem_indices = [fmem_idx]
        for tensor in self.input_tensors[1:]:
            fidx, ex = self.get_fmem_info(in_x, tensor)
            if ex != effective_x:
                raise RuntimeError("Location must be same (Hardware limitation)")
            fmem_indices.append(fidx)
        self.logger.debug(fmem_indices)
        for y in range(in_h):
            fmem_start_row = (
                self.input_tensor.get_address((effective_x, y, z_pivot)) // self.system_width
            )
            for row in range(z_pivot, z_pivot + z_iter):  # I'm not sure that input virtualization can be applied on arithmetic operators
                z = row * self.system_width
                self.output_loc = (x, y, z)
                for i, fidx in enumerate(fmem_indices):
                    yield generate_dataflow_info(
                        phase=1,
                        loc=self.output_loc,
                        reset=(i==0),
                        last=(i==len(fmem_indices)-1),
                        fmem_idx=fidx,
                        fmem_row=fmem_start_row + row,
                        wmem_row=i,
                    )

    def rope_worker(self, x, z_pivot, z_iter, *args, **kwargs):
        main_op = self.main_op
        _, in_h, in_c = self.shape
        half_c = in_c // 2
        half_z_iter = z_iter // 2
        assert half_c % self.system_width == 0
        assert in_c // self.system_width == z_iter
        in_x, _, _ = self.input_tensor.get_loc((x, 0, 0))
        fmem_idx, effective_x = self.get_fmem_info(in_x)
        for y in range(in_h):   # should be 1
            fmem_start_row = (
                self.input_tensor.get_address((effective_x, y, z_pivot)) // self.system_width
            )
            wmem_start_row = (
                self.input_tensors[-1].get_address((in_x, 0, z_pivot)) // self.system_width
            )
            for row in range(z_pivot, z_pivot + z_iter):
                z = row * self.system_width
                self.output_loc = (x, y, z)
                row_front = row % half_z_iter
                row_rear = row_front + half_z_iter
                yield generate_dataflow_info(
                    phase=1,
                    loc=self.output_loc,
                    reset=True,
                    last=False,
                    fmem_idx=fmem_idx,
                    fmem_row=fmem_start_row + row_front,
                    wmem_row=wmem_start_row + row,
                )
                yield generate_dataflow_info(
                    phase=1,
                    loc=self.output_loc,
                    reset=False,
                    last=True,
                    fmem_idx=fmem_idx,
                    fmem_row=fmem_start_row + row_rear,
                    wmem_row=wmem_start_row + z_iter + row,
                )

    def reduction_worker(self):
        filter_idx = 0
        while filter_idx < self.output_shape[-1]:
            self.output_loc = (0, 0, filter_idx)
            yield generate_dataflow_info(phase=2, loc=self.output_loc, last=True)
            filter_idx += self.system_width


class ControlLogicLv1(ControlLogicLv0):  # Only generate 1 signal per output pixel
    def working(self, worker):
        for dataflow in worker:
            if dataflow.phase in [0, 1, 2] and not dataflow.last:
                self.skipped_cycles += 1
                if dataflow.phase == 1:  # Update statistics
                    if dataflow.fmem_row >= 0:
                        self.manager.stats.read_fmem()
                    if dataflow.wmem_row >= 0:
                        self.manager.stats.read_wmem()
            else:
                yield dataflow


