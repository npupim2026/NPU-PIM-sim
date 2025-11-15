from collections import OrderedDict

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from data_structure.attrdict import AttrDict
from software.generic_op import *
from config import cfg

# Note that this builder does not support weight initialization


def get_padding(kernel, dilation, option):
    if isinstance(option, int):
        return (option, option, option, option)
    if isinstance(option, tuple) or isinstance(option, list):
        if len(option) == 4:
            return option
        elif len(option) == 2:
            return (option[0], option[0], option[1], option[1])
        else:
            raise ValueError("Invalid padding: {}".format(option))
    if isinstance(option, str):
        if option.lower() == 'valid':
            return (0, 0, 0, 0)
        if option.lower() == 'same':
            if isinstance(kernel, int):
                k = (kernel, kernel)
            else:
                k = kernel
            pad_t = (k[0] - 1) // 2
            pad_b = k[0] - 1 - pad_t
            pad_l = (k[1] - 1) // 2
            pad_r = k[1] - 1 - pad_l
            return (pad_t * dilation, pad_b * dilation, pad_l * dilation, pad_r * dilation)
    raise ValueError("Invalid padding: {}".format(option))


class ModelBuilder(object):
    def __init__(self, name="custom_model"):
        self.model_dict = OrderedDict()
        self.name_gen = {}
        self.name = name

    def __del__(self):
        del self.model_dict, self.name_gen, self.name

    def set_input_tensor(self, name='input', tensor_shape=(1, 3, 224, 224), input_tensor=None, order='NCHW'):
        if input_tensor is None:
            if tensor_shape[0] == 1:
                tensor_shape = (cfg.MODEL.NUM_FRAMES, *tensor_shape[1:])
            x = torch.randn(*tensor_shape, requires_grad=False)
            input_tensor = x.detach().numpy()
        else:
            if order == 'WHC':
                input_tensor = input_tensor.transpose(2, 1, 0)
                input_tensor = input_tensor[np.newaxis, :]
            elif order == 'NCHW':
                pass
            else:
                raise ValueError("Unknown input dimension, it should be one of ['NCHW', 'WHC']")
            x = torch.from_numpy(input_tensor)
        data = DummyOp(name=name, op_type='HEAD',
                       output_tensor=input_tensor)
        self._add_model(x, data)
        return name

    def get_operator_dict(self):
        ret = OrderedDict()
        for key in self.model_dict:
            ret[key] = self.model_dict[key].generic
        return ret

    def _get_name(self, pre_name, name):
        if name is not None:
            return name
        if pre_name in self.name_gen:
            self.name_gen[pre_name] += 1
        else:
            self.name_gen[pre_name] = 1
        return pre_name + str(self.name_gen[pre_name])

    def _add_model(self, output, generic_op):
        self.model_dict[generic_op.name] = AttrDict({'output': output, 'generic': generic_op})

    def _get_act(self, activation):
        if activation is None:
            activation = 'linear'
        activation = activation.lower()
        torch_act = nn.Identity()
        if activation == 'relu':
            torch_act = nn.ReLU()
        elif activation == 'relu6':
            torch_act = nn.ReLU6()
        elif activation == 'leakyrelu':
            torch_act = nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'sigmoid':
            torch_act = nn.Sigmoid()
        elif activation == 'gelu':
            torch_act = nn.GELU()
        else:
            activation = None
        return activation, torch_act

    def _get_data_with_pad(self, input_name, pad=None, replicate = False):
        if pad is None or pad == (0, 0, 0, 0):
            return self.model_dict[input_name].output
        else:
            pad_t, pad_b, pad_l, pad_r = pad
            if not replicate:
                return nn.ZeroPad2d((pad_l, pad_r, pad_t, pad_b))(self.model_dict[input_name].output)
            else:
                return nn.ReplicationPad2d((pad_l, pad_r, pad_t, pad_b))(self.model_dict[input_name].output)

    def DummyNode(self, input_name, name=None):
        name = self._get_name(input_name + '_', name)
        input_data = self.model_dict[input_name].output
        output = input_data
        generic_op = DummyOp(name=name, input_layers=[input_name], op_type='BYPASS', output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def HostProcessNode(self, inputs, op_call, name=None, mapping='cpu'):
        name = self._get_name('HostProcess', name)
        if isinstance(inputs, str):
            inputs = [inputs]
        input_data = [self.model_dict[input_name].output for input_name in inputs]
        output = op_call(*input_data)
        generic_op = HostProcessOp(name=name, input_layers=inputs, output_tensor=output.detach().numpy(), mapping=mapping)
        self._add_model(output, generic_op)
        return name

    def Constant(self, tensor_shape=(1, 3, 224, 224), data=None, name=None, order='NCHW'):
        name = self._get_name('Constant', name)
        name = self.set_input_tensor(name=name, tensor_shape=tensor_shape, input_tensor=data, order=order)
        self.model_dict[name].generic.type = 'CONST' # Almost same w/ input tensors except operation type
        return name

    def Conv(self, input_name, in_c, out_c, k, stride=1, pad=0, dilation=1, groups=1, bias=True, activation='Relu', name=None):
        name = self._get_name('Conv', name)
        pad = get_padding(k, dilation, pad)
        input_data = self._get_data_with_pad(input_name, pad)
        torch_conv = nn.Conv2d(in_c, out_c, k, stride, 0,
                               dilation, groups, bias, padding_mode='zeros')
        activation, torch_act = self._get_act(activation)
        torch_layer = nn.Sequential(torch_conv, torch_act)
        output = torch_layer(input_data)
        generic_op = ConvOp(
            name=name,
            input_layers=[input_name],
            weight=torch_conv.weight.detach().numpy(),
            bias=torch_conv.bias.detach().numpy() if bias else None,
            dilation=dilation,
            group=groups,
            kernel=k,
            stride=stride,
            pad=pad,
            output_tensor=output.detach().numpy(),
            activation=activation
        )
        self._add_model(output, generic_op)
        return name

    def F_Conv(
            self,
            input_name,
            weight=None,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bias=None,
            activation='Relu',
            order='NCHW',
            name=None):
        name = self._get_name('Conv', name)
        if weight is None:
            raise ValueError("weight data is required")
        if order != 'NCHW':
            raise ValueError("NCHW Input is only supported vs. {}".format(order))
        n, c, k_h, k_w = weight.shape
        from copy import copy
        weight_tensor = torch.from_numpy(copy(weight))
        bias_tensor = None if bias is None else torch.from_numpy(bias)
        k = [k_h, k_w]
        pad = get_padding(k, dilation, pad)
        input_data = self._get_data_with_pad(input_name, pad)
        # print('input shape : {}'.format(input_data.shape))
        conv_output = F.conv2d(input_data, weight_tensor, bias_tensor, stride, 0, dilation, groups)
        activation, torch_act = self._get_act(activation)
        torch_layer = nn.Sequential(torch_act)
        output = torch_layer(conv_output)
        # print('output shape: {}'.format(output.shape))
        generic_op = ConvOp(
            name=name,
            input_layers=[input_name],
            weight=weight,
            bias=bias,
            dilation=dilation,
            group=groups,
            kernel=k,
            stride=stride,
            pad=pad,
            output_tensor=output.detach().numpy(),
            activation=activation
        )
        self._add_model(output, generic_op)
        return name

    def DWConv(self, input_name, in_c, k, stride=1, pad=0, dilation=1, bias=True, activation='Relu', name=None):
        name = self._get_name('DWConv', name)
        return self.Conv(input_name, in_c, in_c, k, stride, pad, dilation, in_c, bias, activation, name)

    def FC(self, input_name, in_c, out_c, bias=True, activation='Linear', name=None):
        name = self._get_name('Linear', name)
        return self.Conv(input_name, in_c, out_c, k=1, bias=bias, activation=activation, name=name)

    def _MatMul_const(self, input_name, M, K, N, activation='Linear', name=None):
        return self.Conv(input_name, K, N, k=1, bias=False, activation=activation, name=name)

    def _MatMul_binary(self, a, b, M, K, N, activation='Linear', name=None):
        # XXX: This op. may be not fully supported without additional HW support
        mat_a = self.model_dict[a].output
        mat_b = self.model_dict[b].output
        mat_a = torch.transpose(mat_a.view(mat_a.shape[1], -1), 0, 1)
        mat_b = torch.transpose(mat_b.view(mat_b.shape[1], -1), 0, 1)

        output = torch.transpose(torch.matmul(mat_a, mat_b), 0, 1)
        output = output.view(1, output.shape[0], -1, 1)

        activation, torch_act = self._get_act(activation)
        output = torch_act(output)

        generic_op = MatMulOp(
            name=name,
            op_type='MatMul',
            input_layers=[a, b],
            output_tensor=output.detach().numpy(),
            vec_len = K,
            activation=activation,
        )
        self._add_model(output, generic_op)
        return name

    def MatMul(self, input_layers, M, K, N, activation='Linear', name=None):
        name = self._get_name('MatMul', name)
        if len(input_layers) == 1:
            return self._MatMul_const(input_layers[0], M, K, N, activation=activation, name=name)
        elif len(input_layers) == 2:
            return self._MatMul_binary(input_layers[0], input_layers[1], M, K, N, activation=activation, name=name)

    def MatMul_binary_transpose(self, a, b_trans, M, K, N, activation='Linear', name=None):
        name = self._get_name('MatMul', name)
        mat_a = self.model_dict[a].output
        mat_b = self.model_dict[b_trans].output
        mat_a = torch.transpose(mat_a.view(mat_a.shape[1], -1), 0, 1)
        mat_b = mat_b.view(mat_b.shape[1], -1)

        output = torch.transpose(torch.matmul(mat_a, mat_b), 0, 1)
        output = output.view(1, output.shape[0], -1, 1)

        activation, torch_act = self._get_act(activation)
        output = torch_act(output)

        generic_op = MatMulOp(
            name=name,
            op_type='MatMulTrans',
            input_layers=[a, b_trans],
            output_tensor=output.detach().numpy(),
            vec_len = K,
            activation=activation,
        )
        self._add_model(output, generic_op)
        return name

    def LayerNorm(self, input_name, normalized_shape=768, name=None):
        # Assumption: normalize along the last dimension
        name = self._get_name("LayerNorm", name)
        input_data = self.model_dict[input_name].output
        input_data_view = torch.transpose(input_data.view(input_data.shape[1], -1), 0, 1)
        batch_size = input_data_view.shape[-2]
        torch_layer = nn.LayerNorm(normalized_shape=normalized_shape)
        output = torch.transpose(torch_layer(input_data_view), 0, 1)
        output = output.view(1, output.shape[0], -1, 1)
        test_code = [
            ('LOAD', 0, 64),
            ('WEIGHT_LOAD', None, None),
            ('PROCESS', ['W', 0], math.ceil(normalized_shape / 64) * math.ceil(batch_size/16)),   # Phase 1: Find mu
            ('PROCESS', [], math.ceil(normalized_shape / 64) * batch_size),                       # Phase 2
            ('PROCESS', [], math.ceil(normalized_shape / 64) * batch_size),
            ('PROCESS', ['L'], math.ceil(normalized_shape / 64) * math.ceil(batch_size/16)),
            ('WEIGHT_PREFETCH', None, None),
            ('PROCESS', [], math.ceil(normalized_shape / 64) * batch_size)
        ]
        generic_op = TestOp(
            name=name,
            input_layers=[input_name],
            output_tensor=output.detach().numpy(),
            weight=np.ones((math.ceil(batch_size/16)*16, 1, 1, normalized_shape)),   # Fake weight
            activation='sigmoid',   # FIXME: Temporal solution; not actually sigmoid but sqrt
            behavior=test_code
        )
        self._add_model(output, generic_op)
        return name

    def RoPEWeight(self, dim, base=10000, input_pos=0, input_len=1, name=None):
        assert dim % 2 == 0
        name_prefix = f'RoPE_({input_pos}'
        if input_len > 1:
            name_prefix += f'-{input_pos+input_len-1}'
        name_prefix += ')_w'
        name = self._get_name(name_prefix, name)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq.view(-1, 1)
        input_pos_seq = torch.arange(input_pos, input_pos + input_len).view(1, -1)
        freqs = (inv_freq @ input_pos_seq.float()).transpose(0, 1).view(1, -1, 1, dim // 2)
        cos = freqs.cos()
        sin = freqs.sin()
        tensor = torch.cat((torch.cat((cos, sin), dim=-1), torch.cat((-sin, cos), dim=-1)), dim=-2)
        tensor = tensor.permute(0, 3, 1, 2).detach().numpy()
        generic_op = HostProcessOp(name=name, op_type='RoPE_weight',
                     output_tensor=tensor)
        generic_op.rope = {'base': base, 'dim': dim, 'input_pos': input_pos, 'input_len': input_len}
        self._add_model(tensor, generic_op)
        return name

    def RoPE(self, input_name, weight, name=None):
        kwargs = self.model_dict[weight].generic.rope
        input_pos = kwargs['input_pos']
        input_len = kwargs['input_len']
        base = kwargs['base']
        dim = kwargs['dim']
        name_prefix = f'RoPE_({input_pos}'
        if input_len > 1:
            name_prefix += f'-{input_pos+input_len-1}'
        name_prefix += ')'
        name = self._get_name(name_prefix, name)
        input_data = self.model_dict[input_name].output
        # the RoPE implementation from huggingface
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq.view(-1, 1)
        input_pos_seq = torch.arange(input_pos, input_pos + input_len).view(1, -1)
        freqs = (inv_freq @ input_pos_seq.float()).transpose(0, 1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        i = input_data.permute(0, 2, 3, 1).view(-1, dim)
        output = (i * cos) + (rotate_half(i) * sin)
        output = output.transpose(0, 1).view(1, dim, -1, 1)
        #input_pos_seq = torch.arange(input_pos, input_pos + input_len).view(1, 1, 1, -1)
        #torch_layer = torchtune.modules.RotaryPositionalEmbeddings(dim=dim, base=base)
        #output = torch_layer(input_data.permute(0, 3, 2, 1), input_pos=input_pos_seq).permute(0, 3, 2, 1)
        generic_op=RoPEOp(
            name=name,
            op_type='RoPE',
            input_layers=[input_name, weight],
            output_tensor=output.detach().numpy(),
            activation=None,
            dim=dim,
            base=base,
            input_pos=input_pos,
            input_len=input_len,
        )
        self._add_model(output, generic_op)
        return name

    def CacheWrite(self, input_name, max_shape, write_offset, write_shape, cache_tag=None, name=None):
        name = self._get_name('Cache', name)
        if cache_tag is None:
            cache_tag = name
        input_data = self.model_dict[input_name].output
        output = torch.zeros(max_shape)
        output[(...,) + tuple(slice(o, o+s) for o, s in zip(write_offset, write_shape))] = input_data
        output = output.view(-1, max_shape[-3], max_shape[-2] // cfg.MIDAP.WMEM.NUM, cfg.MIDAP.WMEM.NUM, max_shape[-1]).transpose(-3, -2).reshape(*max_shape)
        generic_op = CacheOp(
            name=name,
            op_type='Cache',
            input_layers=[input_name],
            output_tensor=output.detach().numpy(),
            activation=None,
            shape=max_shape[-3:],
            tag=cache_tag,
        )
        self._add_model(output, generic_op)
        return name

    def CacheRead(self, input_name, read_offset, read_shape, name=None):
        name = self._get_name('Cache_read', name)
        input_data = self.model_dict[input_name].output
        oz, oy, ox = read_offset
        sz, sy, sx = read_shape
        crop_x = [ox, ox+sx-input_data.shape[-1]]
        crop_y = [oy, oy+sy-input_data.shape[-2]]
        crop_z = [oz, oz+sz-input_data.shape[-3]]
        output = input_data[..., oz:oz+sz, oy:oy+sy, ox:ox+sx]
        generic_op = Crop(name=name, input_layers=[input_name], output_tensor=output.detach().numpy(), crop_x=crop_x, crop_y=crop_y, crop_z=crop_z)
        self._add_model(output, generic_op)
        return name

    def Concat(self, input_layers, axis=1, output_tensor=None, name=None):
        if isinstance(axis, str):
            axis = axis.lower()
        if axis in ['c', 'z']:
            axis = 1
        elif axis in ['w', 'x']:
            raise ValueError("x-axis concatenation is not supported in MIDAP")
        elif axis in ['h', 'y', 2]:
            axis = 2
        if not isinstance(axis, int) or axis > 3:
            raise ValueError("Unknown axis : {}".format(axis))
        name = self._get_name('Concat', name)
        inputs = [self.model_dict[x].output for x in input_layers]

        # Set PyTorch Output
        if output_tensor is None:
            output = torch.cat(inputs, dim=axis)
            output_tensor = output.detach().numpy()
        else:
            output = torch.from_numpy(output_tensor)

        concat_info = [x.shape[axis] for x in inputs]
        generic_op = ConcatOp(name=name, input_layers=input_layers, axis=axis,
                              concat_info=concat_info, output_tensor=output_tensor)
        self._add_model(output, generic_op)
        return name

    def _Pool(self, input_name, pool_type, k, stride=1, pad=0, in_plane=False, name=None):
        name = self._get_name(pool_type, name)
        pad_orig = pad if not isinstance(pad, str) else get_padding(k, 1, pad)
        pad = get_padding(k, 1, pad)
        if pool_type == 'MaxPool':
            input_data = self._get_data_with_pad(input_name, pad)
            pad = 0
        elif in_plane:
            input_data = self._get_data_with_pad(input_name, pad, replicate = True)
            pad = 0
        elif pad[0] == pad[1] and pad[2] == pad[3]:
            pad = pad_orig = (pad[0], pad[2])
            input_data = self._get_data_with_pad(input_name, None)
        else:
            raise ValueError(
                "uneven padding is not supported on Pooling operation")
        torch_pool = nn.MaxPool2d(k, stride, pad) if pool_type == 'MaxPool' else nn.AvgPool2d(
            k, stride, pad, count_include_pad=False)
        output = torch_pool(input_data)
        generic_op = PoolOp(
            name=name,
            op_type=pool_type,
            input_layers=[input_name],
            kernel=k,
            stride=stride,
            pad=pad_orig,
            in_plane = in_plane,
            output_tensor=output.detach().numpy()
        )
        self._add_model(output, generic_op)
        return name

    def MaxPool(self, input_name, k, stride=1, pad=0, name=None):
        return self._Pool(input_name, 'MaxPool', k, stride, pad, False, name)

    def AvgPool(self, input_name, k, stride=1, pad=0, in_plane = False, name=None):
        return self._Pool(input_name, 'AveragePool', k, stride, pad, in_plane, name)

    def GlobalPool(self, input_name, name=None):
        name = self._get_name('GlobalPool', name)
        input_data = self.model_dict[input_name].output
        output = torch.nn.AdaptiveAvgPool2d(1)(input_data)
        generic_op = PoolOp(
            name=name,
            op_type='AveragePool',
            input_layers=[input_name],
            kernel=1,
            stride=1,
            pad=0,
            global_pooling=True,
            output_tensor=output.detach().numpy()
        )
        self._add_model(output, generic_op)
        return name

    def Upsample(self, input_name, scale = None, size = None, algorithm='NN', name=None):
        if size is not None:
            input_data = self.model_dict[input_name].output
            in_hw = input_data.shape[2:]
            out_hw = size
            if out_hw[0] % in_hw[0] != 0 or out_hw[1] % in_hw[1] != 0:
                raise ValueError("support scale with natural number")
            scale = tuple(out_hw[i]//in_hw[i] for i in range(2))
        if algorithm in ['NN', 'nearest']:
            return self.UpsampleNN(input_name, scale, name)
        if algorithm == 'bilinear':
            return self.UpsampleBilinear(input_name, scale, name=name)
    
    def UpsampleNN(self, input_name, scale, name=None):
        name = self._get_name('UpsampleNN', name)
        input_data = self.model_dict[input_name].output
        output = torch.nn.UpsamplingNearest2d(scale_factor=scale)(input_data)
        generic_op = UpsampleOp(name=name, input_layers=[
                                input_name], kernel=scale, algorithm='NN', output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def UpsampleBilinear(self, input_name, scale, name=None):
        return self._UpsampleBilinear_v2(input_name, scale, name)
    
    def _UpsampleBilinear_v1(self, input_name, scale, name=None):
        name = self._get_name('UpsampleNN', name)
        input_data = self.model_dict[input_name].output
        output = torch.nn.UpsamplingNearest2d(scale_factor=2*scale)(input_data)
        generic_op = UpsampleOp(name=name, input_layers=[
                                input_name], kernel=2*scale, algorithm='NN', output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        input_name = name
        name = None
        avg_k = 2*scale
        avg_s = 2
        avg_pad = scale-1
        # avg_pad = 
        return self.AvgPool(input_name, avg_k, avg_s, avg_pad, in_plane=True)

    def _UpsampleBilinear_v2(self, input_name, scale, name=None):
        input_data = self.model_dict[input_name].output
        output = torch.nn.Upsample(scale_factor=scale, mode='bilinear')(input_data)
        if scale > 1:
            name = self._get_name('UpsampleZero', name)
            w = input_data.new_zeros(scale, scale)
            w[0, 0] = 1
            upsample_output = F.conv_transpose2d(input_data, w.expand(input_data.size(
                1), 1, scale, scale), stride=scale, groups=input_data.size(1))
            # upsample_output = upsample_output[:, :, :-(stride - 1), :-(stride - 1)]
            generic_op = UpsampleOp(name=name, algorithm='Zero', input_layers=[
                                    input_name], kernel=scale, output_tensor=upsample_output.detach().numpy())
            self._add_model(upsample_output, generic_op)
            input_name = name

        name = self._get_name('UpsampleBilinear', None)
        if scale % 2 == 0:  # even case
            weight_size = 2 * scale
            s = 2
        else:               # odd case
            weight_size = 2 * scale - 1
            s = 1
        weight = np.empty((64, 1, weight_size, weight_size))
        ltpad = (scale * 3) // 2 - 1
        rbpad = scale // 2
        pad = (ltpad, rbpad, ltpad, rbpad)
        for i in range(scale):
            for j in range(scale):
                weight_value = (i * s + 1) * (j * s + 1) / (scale * s) ** 2
                weight[:, :, i, j] = weight_value
                weight[:, :, i, -j-1] = weight_value
                weight[:, :, -i-1, j] = weight_value
                weight[:, :, -i-1, -j-1] = weight_value

        generic_op = ConvOp(
            name=name,
            op_type='UpsampleBilinear',
            input_layers=[input_name],
            weight=weight,
            bias=None,
            dilation=1,
            group=input_data.shape[-3],
            kernel=weight_size,
            stride=1,
            pad=pad,
            output_tensor=output.detach().numpy(),
            in_plane=True,
            activation=None
        )
        self._add_model(output, generic_op)
        return name

    def Reorg(self, input_name, scale = None, name=None):  ## TODO: Compiler should be implemented
        name = self._get_name('PixelShuffle', name)
        input_data = self.model_dict[input_name].output
        output = torch.nn.PixelShuffle(scale)(input_data)
        generic_op = ReorgOp(name=name, input_layers=[
                                input_name], kernel=scale, output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def ConvTranspose(self, input_name, in_c, out_c, k, stride=1, pad=0, groups=1, bias=True, dilation=1, activation='ReLU', name=None):
        input_data = self.model_dict[input_name].output
        if stride > 1:
            name = self._get_name('UpsampleZero', name)
            w = input_data.new_zeros(stride, stride)
            w[0, 0] = 1
            upsample_output = F.conv_transpose2d(input_data, w.expand(input_data.size(
                1), 1, stride, stride), stride=stride, groups=input_data.size(1))
            # upsample_output = upsample_output[:, :, :-(stride - 1), :-(stride - 1)]
            generic_op = UpsampleOp(name=name, algorithm='Zero', input_layers=[
                                    input_name], kernel=stride, output_tensor=upsample_output.detach().numpy())
            self._add_model(upsample_output, generic_op)
            input_name = name
            name = None
        if not isinstance(pad, str):
            pad = [pad for _ in range(2)] if isinstance(pad, int) else pad
            k = [k, k] if isinstance(k, int) else k
            pad_t = k[0] - 1 - pad[0]
            pad_b = k[0] - 1 - pad[0] - (stride - 1)
            pad_l = k[1] - 1 - pad[1]
            pad_r = k[1] - 1 - pad[1] - (stride - 1)
            pad = (pad_t, pad_b, pad_l, pad_r)
        return self.Conv(input_name, in_c, out_c, k, 1, pad, 1, groups, activation=activation, bias=bias, name=name)

    def Sum(self, input1, input2, activation='Linear', name=None):
        name = self._get_name('Sum', name)
        inputd1, inputd2 = [
            self.model_dict[x].output for x in [input1, input2]]
        output = torch.add(inputd1, inputd2)
        broadcast = inputd1.shape != inputd2.shape
        activation, torch_act = self._get_act(activation)
        output = torch_act(output)
        generic_op = SumOp(name=name, input_layers=[
                           input1, input2], activation=activation, broadcast = broadcast,
                           output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def Mul(self, input1, input2, name=None):
        name = self._get_name('Mul', name)
        inputd1, inputd2 = [
            self.model_dict[x].output for x in [input1, input2]]
        output = torch.mul(inputd1, inputd2)
        broadcast = inputd1.shape != inputd2.shape
        # activation, torch_act = self.get_act(activation)
        # output = torch_act(output)
        generic_op = MulOp(name=name, input_layers=[input1, input2], broadcast = broadcast,
        output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def Crop(self, input1, crop_x=None, crop_y=None, name=None):
        name = self._get_name('Crop', name)
        inputd1 = self.model_dict[input1].output
        output = inputd1.detach().numpy()
        if crop_x is not None:
            x1, x2 = crop_x
            x2 = x2 if x2 <= 0 else x2 - output.shape[-1]
            output = output[:, :, :, x1:output.shape[-1] + x2]
            crop_x = [x1, x2]
        if crop_y is not None:
            y1, y2 = crop_y
            y2 = y2 if y2 <= 0 else y2 - output.shape[-2]
            crop_y = [y1, y2]
            output = output[:, :, y1:output.shape[-2] + y2, :]
        generic_op = Crop(name=name, input_layers=[input1], crop_x=crop_x, crop_y=crop_y, output_tensor=output)
        self._add_model(torch.from_numpy(output), generic_op)
        return name

    def Split(self, input_name, split_size_or_section, dim=1, name=None):
        input_data = self.model_dict[input_name].output
        output_data = input_data.split(split_size_or_section, dim)
        kwarg_key = {1:'crop_z', 2:'crop_y', 3:'crop_x'}[dim]
        if isinstance(split_size_or_section, int):
            size = split_size_or_section
            start_pos = [i for i in range(0, input_data.shape[dim], size)]
        else:
            start_pos = [0]
            for item in split_size_or_section[:-1]:
                start_pos.append(start_pos[-1] + item)
        name_prefix = self._get_name('Split', name)
        name_list = [f'{name_prefix}_{i}' for i in range(len(output_data))]
        for name, start, t in zip(name_list, start_pos, output_data):
            generic_op = Crop(name=name, input_layers=[input_name], output_tensor=t.detach().numpy(), **{kwarg_key: [start, start + t.shape[dim] - input_data.shape[dim]]})
            self._add_model(t, generic_op)
        return name_list

    def Softmax(self, input1, name=None):
        name = self._get_name('Softmax', name)
        inputd1 = self.model_dict[input1].output
        output = torch.nn.Softmax2d()(inputd1)
        generic_op = OperatorBase(name=name, op_type='Softmax', input_layers=[
                                  input1], output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def from_generic_op(self, op, new_input=None, name=None):
        if new_input is None:
            new_input = op.input_layers
            name = op.name
        if not isinstance(op, OperatorBase):
            raise ValueError("op must be defined as generic_op")
        print('convert op: {}'.format(op))
        if isinstance(op, ConvOp):
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            return self.from_conv_op(input_name, op, name)
        elif isinstance(op, ConcatOp):
            return self.Concat(new_input, op.axis, name)
        elif isinstance(op, PoolOp):
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            return self.from_pool_op(input_name, op, name)
        elif isinstance(op, UpsampleOp):
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            return self.Upsample(input_name, op.k_h, op.algorithm, name)
        elif isinstance(op, ArithmeticOp):
            if isinstance(new_input, list) and len(new_input) == 2:
                pass
            else:
                raise ValueError("ArithmeticOp input must be a length-2 list")
            input1, input2 = new_input
            if isinstance(op, SumOp):
                return self.Sum(input1, input2, op.activation, name)
            elif isinstance(op, MulOp):
                return self.Mul(input1, input2, op.activation, name)
            else:
                raise ValueError("Unknown ArithmeticOp: {}".format(op))
        elif isinstance(op, Crop):
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            crop_x = op.crop_x
            crop_y = op.crop_y
            return self.Crop(input_name, crop_x, crop_y, name)
        elif op.type == 'Softmax':
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            return self.Softmax(input_name, name)
        else:
            raise ValueError("Unknown Operator: {}".format(op))

    def from_conv_op(self, input_name, conv_op, pad=None, name=None):
        weight = conv_op.weight_origin
        order = conv_op.order
        # print('weight.shape: {}'.format(weight.shape))
        # print('order: {}'.format(order))
        if order == 'NWHC':
            weight = weight.transpose(0, 3, 2, 1)
            # print('transposed weight.shape: {}'.format(weight.shape))
        stride = conv_op.stride
        if not pad:
            pad = [conv_op.pad_t, conv_op.pad_b, conv_op.pad_l, conv_op.pad_r]
        dilation = conv_op.dilation
        groups = weight.shape[0] if conv_op.type == 'Depthwise' else 1
        bias = conv_op.bias_origin
        return self.F_Conv(
            input_name,
            weight,
            stride,
            pad,
            dilation,
            groups,
            bias,
            conv_op.activation,
            'NCHW',
            name
        )

    def from_pool_op(self, input_name, pool_op, pad=None, name=None):
        if pool_op.global_pooling:
            return self.GlobalPool(input_name, name)
        pool_type = pool_op.type
        k = [pool_op.k_h, pool_op.k_w]
        if not pad:
            pad = [pool_op.pad_t, pool_op.pad_b, pool_op.pad_l, pool_op.pad_r]
        stride = pool_op.stride
        return self._Pool(input_name, pool_type, k, stride, pad, name)
