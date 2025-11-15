from __future__ import absolute_import, division, print_function, unicode_literals
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bq, sb, s, n = 8):
        x = input
        if bq is not None:
            bias = bq.mul(sb).floor_()
            # print(bias.shape)
            x = x + bias
        x = x.mul(s).floor_()

        minimum = -(2 ** (n-1))
        maximum = (2 ** (n-1)) - 1
        out = x.clamp(min=minimum, max=maximum)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class Q_Identity(nn.Module):
    def __init__(self):
        super(Q_Identity, self).__init__()
        self.scale = 1.0

    def forward(self, x, n = 8):
        return Quantizer.apply(x, None, 1.0, self.scale, n)

    def build_lut(self, in_scale, out_scale, x_bits, x1_bits):
        pass

class Q_ReLU(nn.Module):
    def __init__(self, inplace=True):
        super(Q_ReLU, self).__init__()
        self.inplace = inplace
        self.scale = 1.0

    def forward(self, x, n = 8):
        x = F.relu(x, self.inplace)
        return Quantizer.apply(x, None, 1.0, self.scale, n)

    def build_lut(self, in_scale, out_scale, x_bits, x1_bits):
        pass

class Q_LeakyReLU(nn.Module):
    def __init__(self, inplace=True, negative_slope = 0.1):
        super(Q_LeakyReLU, self).__init__()
        self.inplace = inplace
        self.scale = 1.0
        self.negative_slope = 0.1
        self.lut = None
        self.lut_embedding = None
        self.div = 1

    def forward(self, x, n = 8):
        x1 = x / self.div
        x1 = x1.floor_()
        x2 = x - (x1 * self.div)
        x1 = x1.long()
        lookup_result = self.lut_embedding[x1, :]
        y = (lookup_result[:,:,:,:,0] * x2 / self.div) + lookup_result[:,:,:,:,1]
        y = y.floor_()
        return Quantizer.apply(y, None, 1.0, self.scale, n)

    def build_lut(self, in_scale, out_scale, x_bits, x1_bits):
        self.lut = list()
        x2_bits = x_bits - x1_bits
        mul = 2 ** x2_bits
        self.div = mul
        for i in range(0, 2 ** (x2_bits - 1)):
            x0_value = i * mul * in_scale
            x1_value = (i+1) * mul * in_scale
            y0 = x0_value if x0_value > 0 else self.negative_slope * x0_value
            y0 = y0 / out_scale
            y1 = x1_value if x1_value > 0 else self.negative_slope * x1_value
            y1 = y1 / out_scale
            b = round(y0)
            a = round(y1 - y0)
            self.lut.append([a, b])
        for i in range(-2 ** (x2_bits - 1), 0):
            x0_value = i * mul * in_scale
            x1_value = (i+1) * mul * in_scale
            y0 = x0_value if x0_value > 0 else self.negative_slope * x0_value
            y0 = y0 / out_scale
            y1 = x1_value if x1_value > 0 else self.negative_slope * x1_value
            y1 = y1 / out_scale
            b = round(y0)
            a = round(y1 - y0)
            self.lut.append([a, b])
        self.lut_embedding = torch.FloatTensor(self.lut)

class Q_Sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Q_Sigmoid, self).__init__()
        self.inplace = inplace
        self.scale = 1.0
        self.lut = None
        self.lut_embedding = None
        self.div = 1

    def forward(self, x, n = 8):
        # x = torch.minimum(torch.tensor([127]), torch.maximum(torch.tensor([0]), x + 64))
        x1 = x / self.div
        x1 = x1.floor_()
        x2 = x - (x1 * self.div)
        x1 = x1.long()
        lookup_result = self.lut_embedding[x1, :]
        y = (lookup_result[:,:,:,:,0] * x2 / self.div) + lookup_result[:,:,:,:,1]
        y = y.floor_()
        return Quantizer.apply(y, None, 1.0, self.scale, n)
    
    def build_lut(self, in_scale, out_scale, x_bits, x1_bits):
        import math
        self.lut = list()
        x2_bits = x_bits - x1_bits
        mul = 2 ** x2_bits
        self.div = mul
        for i in range(0, 2 ** (x2_bits - 1)):
            x0_value = i * mul * in_scale
            x1_value = (i+1) * mul * in_scale
            y0 = (1 / (1 + math.exp(-x0_value))) / out_scale
            y1 = (1 / (1 + math.exp(-x1_value))) / out_scale
            b = round(y0)
            a = round(y1 - y0)
            self.lut.append([a, b])
        for i in range(-2 ** (x2_bits - 1), 0):
            x0_value = i * mul * in_scale
            x1_value = (i+1) * mul * in_scale
            y0 = (1 / (1 + math.exp(-x0_value))) / out_scale
            y1 = (1 / (1 + math.exp(-x1_value))) / out_scale
            b = round(y0)
            a = round(y1 - y0)
            self.lut.append([a, b])
        self.lut_embedding = torch.FloatTensor(self.lut)

class Q_GELU(nn.Module):
    def __init__(self, inplace=True):
        super(Q_GELU, self).__init__()
        self.inplace = inplace
        self.scale = 1.0
        self.lut = None
        self.lut_embedding = None
        self.div = 1

    def forward(self, x, n = 8):
        x1 = x / self.div
        x1 = x1.floor_()
        x2 = x - (x1 * self.div)
        x1 = x1.long()
        lookup_result = self.lut_embedding[x1, :]
        y = (lookup_result[:,:,:,:,0] * x2 / self.div) + lookup_result[:,:,:,:,1]
        y = y.floor_()
        return Quantizer.apply(y, None, 1.0, self.scale, n)

    def build_lut(self, in_scale, out_scale, x_bits, x1_bits):
        import math
        def gelu(x):
            return (x / 2) * (1 + math.erf(x / math.sqrt(2)))
        self.lut = list()
        x2_bits = x_bits - x1_bits
        mul = 2 ** x2_bits
        self.div = mul
        for i in range(0, 2 ** (x2_bits - 1)):
            x0_value = i * mul * in_scale
            x1_value = (i+1) * mul * in_scale
            y0 = gelu(x0_value) / out_scale
            y1 = gelu(x1_value) / out_scale
            b = round(y0)
            a = round(y1 - y0)
            self.lut.append([a, b])
        for i in range(-2 ** (x2_bits - 1), 0):
            x0_value = i * mul * in_scale
            x1_value = (i+1) * mul * in_scale
            y0 = gelu(x0_value) / out_scale
            y1 = gelu(x1_value) / out_scale
            b = round(y0)
            a = round(y1 - y0)
            self.lut.append([a, b])
        self.lut_embedding = torch.FloatTensor(self.lut)

class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.scale = 1.0
        self.bias_scale = 1.0

    def initialize(self, s, sb):
        self.scale = s
        self.bias_scale = sb

    def forward(self, x, n = 16):
        y = F.conv2d(
            x, self.weight, None, self.stride, self.padding, self.dilation, self.groups
        )
        return Quantizer.apply(y, self.bias, self.bias_scale, self.scale, n)

class Q_Linear(nn.Linear):
    def __init__(self, *args, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.scale = 1.0
        self.bias_scale = 1.0

    def initialize(self, s, sb):
        self.scale = s
        self.bias_scale = sb

    def forward(self, x, n = 16):
        y = F.linear(x, self.weight, None)
        return Quantizer.apply(y, self.bias, self.bias_scale, self.scale, n)

class Q_MatMul_transpose(nn.Module):
    def __init__(self, *args, **kargs):
        super(Q_MatMul_transpose, self).__init__(*args, **kargs)
        self.scale = 1.0

    def initialize(self, s, *args, **kwargs):
        self.scale = s

    def forward(self, a, b_trans, n = 16):
        mat_a = torch.transpose(a.view(a.shape[1], -1), 0, 1)
        mat_b = b_trans.view(b_trans.shape[1], -1)

        mat_c = torch.transpose(torch.matmul(mat_a, mat_b), 0, 1)
        mat_c = mat_c.view(1, mat_c.shape[0], 1, -1)

        return Quantizer.apply(mat_c, None, 1.0, self.scale, n)

class Q_Add(nn.Module):
    def __init__(self, *args, **kargs):
        super(Q_Add, self).__init__(*args, **kargs)
        self.scale = 1.0

    def initialize(self, s, *args, **kwargs):
        self.scale = s

    def forward(self, x1, x2, n = 16):
        y = x1 + x2
        return Quantizer.apply(y, None, 1.0, self.scale, n)

class Q_WeightedSum(nn.Module):
    def __init__(self, weight, *args, **kargs):
        super(Q_WeightedSum, self).__init__(*args, **kargs)
        self.scale = 1.0
        self.weight = weight

    def initialize(self, s, *args, **kwargs):
        self.scale = s

    def forward(self, x1, x2, n = 16):
        y = self.weight[0] * x1 + self.weight[1] * x2
        return Quantizer.apply(y, None, 1.0, self.scale, n)

class Q_Mul(nn.Module):
    def __init__(self, *args, **kargs):
        super(Q_Mul, self).__init__(*args, **kargs)
        self.scale = 1.0

    def initialize(self, s, *args, **kwargs):
        self.scale = s

    def forward(self, x1, x2, n = 16):
        y = x1 * x2
        return Quantizer.apply(y, None, 1.0, self.scale, n)


class Q_AvgPool2d(nn.AvgPool2d):
    def __init__(self, *args, **kargs):
        super(Q_AvgPool2d, self).__init__(*args, **kargs)
        self.scale = 1.0
        self.divisor_override = 1

    def initialize(self, s, *args, **kwargs):
        self.scale = s

    def forward(self, x, n = 16):
        y = super(Q_AvgPool2d, self).forward(x)
        return Quantizer.apply(y, None, 1.0, self.scale, n)


class Q_MaxPool2d(nn.MaxPool2d):
    def __init__(self, *args, **kargs):
        super(Q_MaxPool2d, self).__init__(*args, **kargs)
        self.scale = 1.0

    def initialize(self, s, *args, **kwargs):
        self.scale = s

    def forward(self, x, n = 16):
        y = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        return Quantizer.apply(y, None, 1.0, self.scale, n)


class Q_UpsampleBilinear(nn.Module):
    def __init__(self, scale_factor, *args, **kargs):
        super(Q_UpsampleBilinear, self).__init__(*args, **kargs)
        self.scale_factor = scale_factor
        self.scale = 1.0

    def initialize(self, s, *args, **kwargs):
        self.scale = s

    def forward(self, x, n = 16):
        k = 2 * self.scale_factor - (self.scale_factor % 2)
        ltpad = (self.scale_factor * 3) // 2 - 1
        rbpad = self.scale_factor // 2
        x1 = F.max_pool2d(x[:, :, ltpad:-rbpad, ltpad:-rbpad], self.scale_factor) # cancel padding & zero upsampling
        y = F.interpolate(x1, scale_factor=self.scale_factor, mode='bilinear') * (128 if self.scale_factor % 2 == 0 else 64)
        return Quantizer.apply(y, None, 1.0, self.scale, n)


class Q_RoPE(nn.Module):
    def __init__(self, *args, **kargs):
        super(Q_RoPE, self).__init__(*args, **kargs)
        self.scale = 1.0

    def initialize(self, s, *args, **kwargs):
        self.scale = s

    def forward(self, x, weight, n = 16):
        dim = weight.shape[1]
        x = x.transpose(1, 3)
        x = x.view(x.shape[0], -1, 2, dim//2)
        weight = weight.transpose(1, 3)
        y = (torch.cat((x, x), dim=-1) * weight).sum(dim=2, keepdim=True)
        y = y.transpose(1, 3)
        return Quantizer.apply(y, None, 1.0, self.scale, n)


class QuantOps(object):
    Conv2d = Q_Conv2d
    MaxPool2d = Q_MaxPool2d
    AvgPool2d = Q_AvgPool2d
    MatMulTrans = Q_MatMul_transpose
    Add = Q_Add
    WeightedSum = Q_WeightedSum
    Mul = Q_Mul
    UpsampleBilinear = Q_UpsampleBilinear
    RoPE = Q_RoPE
    ReLU = Q_ReLU
    LeakyReLU = Q_LeakyReLU
    Sigmoid = Q_Sigmoid
    GELU = Q_GELU
    Linear = Q_Linear
    Identity = Q_Identity
