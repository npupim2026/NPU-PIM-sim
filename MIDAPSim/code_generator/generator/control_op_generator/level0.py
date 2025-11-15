import math

from code_generator.env import rule

from .base import ControlOpGenerator

from midap_backend.wrapper.op_wrapper import ConvWrapper

from code_generator.env.sfr import reg
from code_generator.env.opcode import set_reg, reset_reg, wait_fin
from code_generator.env.rule import TriggerType


import numpy as np



class ControlOpGeneratorLv0(ControlOpGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_kernel(self, kernel_size, dilation, pivot_idx, max_idx, axis = 0):
        scale = 1
        if self.input_tensor.is_valid_type:
            scale = self.input_tensor.scale[axis]
        if dilation > 1 and scale > 1:
            raise ValueError("Dilation > 1 && valid input type (VTensor) is not allowed")
        eff_k = kernel_size - 1
        kernel_pivot = 0
        feature_pivot = pivot_idx
        feature_pivot_last = min(pivot_idx + dilation * eff_k, max_idx - 1)
        if pivot_idx < 0:
            kernel_pivot = -1 * (pivot_idx // dilation)
            feature_pivot += kernel_pivot * dilation
        ##
        div = dilation
        if scale > 1:
            new_feature_pivot = scale * math.ceil(feature_pivot / scale)
            feature_pivot_last = scale * (feature_pivot_last // scale)
            kernel_pivot += new_feature_pivot - feature_pivot
            feature_pivot = new_feature_pivot
            div = scale
            self.logger.debug(f"Input: {(kernel_size, dilation, pivot_idx, max_idx, axis)}")
            self.logger.debug(f"[kernel_pivot, feature_pivot, feature_pivot_last, div]: {[kernel_pivot, feature_pivot, feature_pivot_last, div]}")
        eff_k = (feature_pivot_last - feature_pivot) // div
        return kernel_pivot, eff_k
    
    # def preprocess_kernel(self, kernel_size, dilation, pivot_idx, max_idx, axis):
    #     eff_k = kernel_size - 1
    #     kernel_pivot = 0
    #     if pivot_idx + dilation * eff_k >= max_idx:
    #         eff_k -= math.ceil((pivot_idx + dilation * eff_k - max_idx + 1)/dilation)
    #     if pivot_idx < 0:
    #         kernel_pivot = -1 * (pivot_idx // dilation)
    #         eff_k -= kernel_pivot
    #     return kernel_pivot, eff_k

    def get_fmem_read_info(self, x, tensor = None):
        if tensor is None:
            input_mapping = self.input_mapping
        else:
            input_mapping = self.get_input_mapping(tensor)
        for fmem_idx, head, tail in input_mapping[self.input_pivot_idx :]:
            if head <= x and x < tail:
                return fmem_idx, x - head, tail

    def get_fmem_pivot_info(self, pivot_loc, eff_kx, dilation):
        real_in_x = pivot_loc[0]
        in_x, in_y, in_z = self.input_tensor.get_loc(pivot_loc)
        fmem_idx, effective_x, tail_x = self.get_fmem_read_info(in_x)
        rfp_pivot = self.memory_op_generator.get_fmem_addr(fmem_idx, self.input_tensor.get_address((effective_x, in_y, in_z)))
        jump_x = 0
        jfp_pivot = -1
        tail_x = tail_x * self.input_tensor.scale[0]
        if tail_x <= real_in_x + eff_kx * dilation:
            jump_x = math.ceil((tail_x - real_in_x)/dilation)
            jump_idx, jump_ex, _ = \
                self.get_fmem_read_info((real_in_x + jump_x * dilation) // self.input_tensor.scale[0])
            jfp_pivot = self.memory_op_generator.get_fmem_addr(jump_idx, self.input_tensor.get_address((jump_ex, in_y, in_z)))
        return rfp_pivot, jump_x - 1, jfp_pivot

    def gen_write_code(self, in_y, pivot_y=None):
        main_op = self.main_op
        wflag, wtp, wfp = self.write_pivot_info
        fyo, tyo, _ = self.write_offset_info
        pad_t = main_op.pad_t
        s = main_op.stride
        oy = (pad_t + in_y) // s
        pivot_oy = (pad_t + pivot_y) // s if pivot_y is not None else 0
        code = self.set_reg(reg.wi, [wflag, wtp + tyo * (oy - pivot_oy), wfp + fyo * (oy - pivot_oy)])
        return code
    
    def trigger_conv_z(
        self,
        in_y,
        foffset,
        woffset,
        iters,
        z_iter,
        eff_kx,
        eff_ky,
        ncr,
        pivot_x,
        pivot_y,
        stride,
        write_y_offset=None
        ):
        code = []
        if self.input_tensor.is_linear_type:
            it = self.input_tensor
            scale = it.scale
            # no dilation
            code += self.set_reg(reg.tvi1, [1, scale[0], scale[1], pivot_x % scale[0]])
            code += self.set_reg(reg.tvi2, [pivot_y % scale[1], stride, 1, 1])
        code += self.gen_write_code(in_y, write_y_offset)
        code += self.set_reg(reg.rso, [foffset, woffset])
        code += self.set_reg(reg.oli, [iters, z_iter - 1]) # Fixed z iterations
        code += self.set_reg(reg.ili, [eff_kx, eff_ky, ncr -1])
        code += self.trigger_and_wait(TriggerType.MAIN)
        return code

    def conv_z_code_generator(
        self,
        in_x,
        filter_idx,
        z_iter,
        head_y,
        tail_y,
        stride,
        write_y_offset=None
    ):
        if write_y_offset is None:
            write_y_offset = head_y
        input_tensor = self.input_tensor
        main_op = self.main_op
        dilation = main_op.dilation
        k_h, k_w = main_op.k_h, main_op.k_w
        in_w, in_h, in_c = input_tensor.shape
        _, _, wio, wjo, z_unit, ncr = self.params
        rpad_l, rpad_r, rpad_t, rpad_b = 0, k_w, 0, k_h
        rpad_flag = 0x2 if main_op.in_plane and (in_x < 0 or in_x + k_w - 1 >= in_w) else 0
        # Get Effective-x Kernel
        kernel_pivot_x, eff_kx = self.preprocess_kernel(k_w, dilation, in_x, in_w, 0)
        in_x += dilation * kernel_pivot_x
        z_offset = 0 if isinstance(main_op, ConvWrapper) else filter_idx
        # Get FMEM Pivot addrs
        rfp_pivot, jump_x, jfp_pivot = \
            self.get_fmem_pivot_info((in_x, 0, z_offset), eff_kx, dilation)
        if main_op.in_plane:
            div = input_tensor.scale[0] if input_tensor.is_valid_type else 1
            rpad_l = kernel_pivot_x // div
            rpad_r = rpad_l + eff_kx
            kernel_pivot_x %= div
            eff_kx = k_w // div - 1
        rwp_pivot = self.memory_op_generator.get_wmem_addr(0)
        rwp_pivot += wio * kernel_pivot_x
        # Generate
        code = []
        code += self.set_reg(reg.rp, [rfp_pivot, rwp_pivot])
        code += self.set_reg(reg.rpad, [rpad_flag, rpad_l, rpad_r, 0, k_h])
        if jfp_pivot >= 0:
            code += self.set_reg(reg.ji, [1, jump_x, jfp_pivot])
        else:
            code += self.set_reg(reg.ji, [0, 0, 0])
        y = head_y
        while y < 0:
            kpy, eff_ky = self.preprocess_kernel(k_h, dilation, y, in_h, 1)
            fpy = y + kpy * dilation
            if main_op.in_plane:
                div = input_tensor.scale[1] if input_tensor.is_valid_type else 1
                rpad_t = kpy // div
                rpad_b = rpad_t + eff_ky
                kpy %= div
                eff_ky = k_h // div - 1
                code += self.set_reg(reg.rpad, [rpad_flag | 0x1, rpad_l, rpad_r, rpad_t, rpad_b])
            foffset = (fpy // input_tensor.scale[1]) * z_unit
            woffset = kpy * wjo
            code += self.trigger_conv_z(
                y,
                foffset,
                woffset,
                0,
                z_iter,
                eff_kx,
                eff_ky,
                ncr,
                in_x,
                fpy,
                stride,
                write_y_offset
                )
            y += stride
        if head_y < 0 and main_op.in_plane:
            code += self.set_reg(reg.rpad, [rpad_flag, rpad_l, rpad_r, 0, k_h])
        kpy, eff_ky = self.preprocess_kernel(k_h, dilation, y, in_h, 1)
        iters = (min(in_h - (k_h-1) * dilation - 1, tail_y) - y) // stride
        # self.logger.info(f"[in_h, k_h, dilation, y, stride] = {[in_h, k_h, dilation, y, stride]}")
        # self.logger.info(f"-> iters : {iters}")
        foffset = ((y + kpy * dilation) // self.input_tensor.scale[1]) * z_unit
        woffset = kpy * wjo
        if iters >= 0: 
            code += self.trigger_conv_z(
                y,
                foffset,
                woffset,
                iters,
                z_iter,
                eff_kx,
                eff_ky,
                ncr,
                in_x,
                y + kpy,
                stride,
                write_y_offset
                )
            y += stride * (iters + 1)
        while y <= tail_y:
            kpy, eff_ky = self.preprocess_kernel(k_h, dilation, y, in_h, 1)
            if main_op.in_plane:
                div = input_tensor.scale[1] if input_tensor.is_valid_type else 1
                rpad_t = kpy // div
                rpad_b = rpad_t + eff_ky
                eff_ky = k_h // div - 1
                code += self.set_reg(reg.rpad, [rpad_flag | 0x1, rpad_l, rpad_r, rpad_t, rpad_b])
            if eff_ky == k_h - 1 and not main_op.in_plane:
                raise ValueError("effective y-kernel must be smaller than k_h - 1.. k_h -1 , eff_ky = [{} vs {}], dilation = {}, y = {}, tail_y = {}, in_h = {}".format(k_h - 1, eff_ky, dilation, y, tail_y, in_h))
            foffset = ((y + kpy * dilation) // self.input_tensor.scale[1]) * z_unit
            woffset = kpy * wjo
            code += self.trigger_conv_z(
                y,
                foffset,
                woffset,
                0,
                z_iter,
                eff_kx,
                eff_ky,
                ncr,
                in_x,
                y + kpy,
                stride,
                write_y_offset
                )
            y += stride
        return code

    def gen_conv_z_code(self, in_x, filter_idx, z_iter, head_y=None, tail_y=None, *args, **kwargs):
        if head_y is None:
            head_y = self.head_y
        if tail_y is None:
            tail_y = self.tail_y
        return self.conv_z_code_generator(
            in_x,
            filter_idx,
            z_iter,
            head_y,
            tail_y,
            self.main_op.stride
        )
    
    def gen_conv_z_tv_code(self, in_x, filter_idx, z_iter, head_y=None, tail_y=None, *args, **kwargs):
        ## Valid input // Linear input
        if head_y is None:
            head_y = self.head_y
        if tail_y is None:
            tail_y = self.tail_y
        code = []
        main_op = self.main_op
        if self.input_tensor.is_valid_type:
            # Do effective kernel computation
            if main_op.stride > 1:
                raise ValueError("Stride > 1 is not allowed for valid type input tensor")
            stride = self.input_tensor.scale[1]
            for i in range(stride):
                code += self.conv_z_code_generator(
                    in_x,
                    filter_idx,
                    z_iter,
                    head_y + i,
                    min(tail_y + i, self.tail_y),
                    stride,
                    head_y
                )
        else:
            if main_op.dilation > 1:
                raise ValueError("Dilation > 1 is not allowed for linear type input tensor")
            code += self.conv_z_code_generator(
                    in_x,
                    filter_idx,
                    z_iter,
                    head_y,
                    tail_y,
                    main_op.stride
                )
        return code

    def trigger_conv_yz(
        self,
        in_y,
        foffset,
        woffset,
        iters,
        z_iter,
        eff_kx,
        nr,
        offset_info,
        write_y_offset=None
        ):
        code = self.gen_write_code(in_y, write_y_offset)
        code += self.set_reg(reg.rso, [foffset, woffset])
        code += self.set_reg(reg.oli, [iters, z_iter - 1]) # Fixed z iterations
        code += self.set_reg(reg.ili, [eff_kx, 0, nr - 1])
        wo, fo, jfo, fio, yo, yzo = offset_info
        offset_info[0] = (wo + woffset) & self.svar.read_offset_mask
        offset_info[1] = (fo + foffset) & self.svar.read_offset_mask
        offset_info[2] = (jfo + foffset) & self.svar.read_offset_mask
        wo, fo, jfo, fio, yo, yzo = offset_info
        eff_last = (wo + yzo) & self.svar.read_offset_mask
        delete_f = np.left_shift(np.left_shift(1, wo) - 1, 4 - wo) # Fixed .. can be gotten from register
        delete_b = (np.left_shift(1, 4 - eff_last) - 1) # Fixed.. can be gotten from register
        if eff_last == 0:
            delete_b = 0
        code += self.set_reg(reg.cyzi, offset_info + [delete_f, delete_b])
        code += self.trigger_and_wait(TriggerType.MAIN)
        return code

    def gen_conv_yz_code(self, in_x, filter_idx, z_iter, head_y=None, tail_y=None, *args, **kwargs):  # default input tensor type
        if head_y is None:
            head_y = self.head_y
        if tail_y is None:
            tail_y = self.tail_y
        main_op = self.main_op
        k_h, k_w = main_op.k_h, main_op.k_w
        s = main_op.stride
        in_w, in_h, in_c = self.input_tensor.shape
        yo, fio, wio, wjo, z_unit, _ = self.params
        # Get Effective-x Kernel
        kernel_pivot_x, eff_kx = self.preprocess_kernel(k_w, 1, in_x, in_w, 0)
        in_x += kernel_pivot_x
        rwp_pivot = self.memory_op_generator.get_wmem_addr(0)
        rwp_pivot += wio * kernel_pivot_x
        # Get FMEM Pivot addrs
        rfp_pivot, jump_x, jfp_pivot = self.get_fmem_pivot_info((in_x, 0, 0), eff_kx, 1)
        # self.save_info = (kernel_pivot_x, eff_kx, effective_x, tail_x, rwp_pivot, rfp_pivot)
        # self.logger.debug("Generator) :in_x: {}, (kpx, ekx, ex, tx, rwpp, rfpp)= {}".format(in_x, self.save_info))
        fo = rfp_pivot & self.svar.read_offset_mask
        fio = fio & self.svar.read_offset_mask
        jfo = jfp_pivot & self.svar.read_offset_mask
        yo = yo & self.svar.read_offset_mask
        wo = rwp_pivot & self.svar.read_offset_mask
        # Generate
        y = head_y
        code = []
        code += self.set_reg(reg.rp, [rfp_pivot, rwp_pivot])
        if jfp_pivot >= 0:
            code += self.set_reg(reg.ji, [1, jump_x, jfp_pivot])
        else:
            code += self.set_reg(reg.ji, [0, 0, 0])
        while y < 0:
            kpy = -y
            yzp = z_unit * (k_h-kpy)
            yzo = yzp & self.svar.read_offset_mask
            nr = math.ceil(k_h * z_unit/self.svar.align) - (kpy * z_unit) // self.svar.align
            code += self.trigger_conv_yz(
                y,
                0,
                kpy * z_unit,
                0,
                z_iter,
                eff_kx,
                nr,
                [wo, fo, jfo, fio, yo, yzo],
                head_y
                )
            y += s

        nr = math.ceil(k_h * z_unit/4)
        eff_ky = k_h - 1
        iters = (min(in_h - eff_ky - 1, tail_y) - y) // s
        if iters >= 0:
            yzp = z_unit * k_h
            yzo = yzp & self.svar.read_offset_mask
            code += self.trigger_conv_yz(
                y,
                y * z_unit,
                0,
                iters,
                z_iter,
                eff_kx,
                nr,
                [wo, fo, jfo, fio, yo, yzo],
                head_y
                )
            y += s * (iters + 1)
        while y <= tail_y:
            kpy, eff_ky = self.preprocess_kernel(k_h, 1, y, in_h, 1)
            if eff_ky == k_h - 1:
                raise ValueError("effective y-kernel must be smaller than k_h - 1.. k_h -1 , eff_ky = [{} vs {}], dilation = {}, y = {}, tail_y = {}, in_h = {}".format(k_h - 1, eff_ky, 1, y, self.tail_y, in_h))
            eff_ky += 1
            nr = math.ceil(eff_ky * z_unit/4)
            yzp = z_unit * eff_ky
            yzo = yzp & self.svar.read_offset_mask
            code += self.trigger_conv_yz(
                y,
                y * z_unit,
                0, 
                0,
                z_iter,
                eff_kx,
                nr,
                [wo, fo, jfo, fio, yo, yzo],
                head_y
                )
            y += s
        return code
    
    def gen_arithmetic_code(self, x, filter_idx, z_iter, head_y=None, tail_y=None, *args, **kwargs):
        if head_y is None:
            head_y = self.head_y
        if tail_y is None:
            tail_y = self.tail_y
        main_op = self.main_op
        fmem_idx, effective_x, _ = self.get_fmem_read_info(x)
        faddr = self.input_tensor.get_address((effective_x, head_y, filter_idx))
        rfp = self.memory_op_generator.get_fmem_addr(fmem_idx, faddr)
        rwp = self.memory_op_generator.get_wmem_addr(filter_idx)    # FIXME: WMEM offset?
        y_iters = tail_y - head_y + 1
        code = []
        code += self.set_reg(reg.wi, self.write_pivot_info)
        code += self.set_reg(reg.rp, [rfp, rwp])
        code += self.set_reg(reg.oli, [y_iters - 1, z_iter - 1]) # Fixed z iterations
        code += self.set_reg(reg.ili, [0, 0, 0])
        code += self.trigger_and_wait(TriggerType.MAIN)
        return code

    def gen_arithmetic_tv_code(self, x, filter_idx, z_iter, head_y=None, tail_y=None, *args, **kwargs):
        raise NotImplementedError("To be determined")
        return []

    def gen_weighted_sum_code(self, x, filter_idx, z_iter, head_y=None, tail_y=None, *args, **kwargs):
        if head_y is None:
            head_y = self.head_y
        if tail_y is None:
            tail_y = self.tail_y
        main_op = self.main_op
        fmem_idx, effective_x, _ = self.get_fmem_read_info(x, self.input_tensor)
        faddr = self.input_tensor.get_address((effective_x, head_y, filter_idx))
        rfp = self.memory_op_generator.get_fmem_addr(fmem_idx, faddr)
        rwp = self.memory_op_generator.get_wmem_addr(filter_idx)
        jump_idx, jump_x, _ = self.get_fmem_read_info(x, self.input_tensors[-1])
        jaddr = self.input_tensors[-1].get_address((jump_x, head_y, filter_idx))
        jfp_pivot = self.memory_op_generator.get_fmem_addr(jump_idx, jaddr)
        y_iters = tail_y - head_y + 1
        code = []
        code += self.set_reg(reg.wi, self.write_pivot_info)
        code += self.set_reg(reg.rp, [rfp, rwp])
        code += self.set_reg(reg.oli, [y_iters - 1, z_iter - 1]) # Fixed z iterations
        code += self.set_reg(reg.ili, [1, 0, 0])
        code += self.set_reg(reg.ji, [1, 0, jfp_pivot])
        code += self.trigger_and_wait(TriggerType.MAIN)
        return code

    def gen_weighted_sum_tv_code(self, x, filter_idx, z_iter, head_y=None, tail_y=None, *args, **kwargs):
        raise NotImplementedError("To be determined")
        return []

    def gen_reduction_code(self):
        # iters = self.output_shape[-1] // self.system_width
        iters = self.output_shape[-1] // self.num_wmem
        code = []
        code += self.set_reg(reg.wi, self.write_pivot_info)
        code += self.set_reg(reg.ili, [0, 0, 0])
        code += self.set_reg(reg.oli, [0, iters - 1])
        code += self.trigger_and_wait(TriggerType.INIT)
        code += wait_fin(rule.PIPELINE_DEPTH)
        code += self._setup_lp(self.layer_info, reduction=True)
        code += self.trigger_and_wait(TriggerType.RD)
        return code
