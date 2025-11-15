from . import ModelBuilder


def se_module(mb : ModelBuilder, name, channel, reduction=16):
    #print("SE_module" + str([name, channel, reduction]))
    avg = mb.GlobalPool(name)
    fc1 = mb.FC(avg, channel, channel // reduction,
                bias=True, activation='Relu')
    fc2 = mb.FC(fc1, channel // reduction, channel,
                bias=True, activation='Sigmoid')
    mul = mb.Mul(name, fc2)
    return mul

def bottleneck_block(mb : ModelBuilder, x, inp, expand = 64, squeeze = 16, se = True, strides = 1, bneck_depth = 3, fused = False, residual = True):
    #print([x, inp, expand, squeeze, se, strides, bneck_depth])
    m = mb.Conv(x, inp, expand, 1, 1)
    if not fused:
        m = mb.DWConv(m, expand, bneck_depth, strides, 'same')
    if se:
        m = se_module(mb, m, expand, 4)
    if not fused:
        m = mb.Conv(m, expand, squeeze, 1, 1)
    else:
        m = mb.Conv(m, expand, squeeze, bneck_depth, strides, 'same')
    if strides == 1 and inp == squeeze and residual:
        m = mb.Sum(x, m)
    return m

def mobilenet_v3(
        version = 'small',
        input_size=224,
        num_outputs = [576, 1024],
        num_classess=1000,
        bneck_info = None,
        classify = False,
        depth_multiplier = 1.0,
        segmentation = False,
        segm_feature_indices = [3, 8],
        *args,
        **kwargs
        ):
    
    # inp channels, exp size, squeeze (output channels), SE, stride, bneck kernel size
    classify = True
    if bneck_info == None:
        bneck_info = [
                [16, 16, 16, True, 2, 3],
                [16, 72, 24, False, 2, 3],
                [24, 88, 24, False, 1, 3],
                [24, 96, 40, True, 2, 5],
                [40, 240, 40, True, 1, 5],
                [40, 240, 40, True, 1, 5],
                [40, 120, 48, True, 1, 5],
                [48, 144, 48, True, 1, 5],
                [48, 288, 96, True, 2, 5],
                [96, 576, 96, True, 1, 5],
                [96, 576, 96, True, 1, 5],
                ]
    for idx in range(len(bneck_info)):
        item = bneck_info[idx]
        for i in range(3):
            item[i] = _make_divisible(item[i] * depth_multiplier, 8, 16)
    mb = ModelBuilder("MobilenetV3_{}_{}".format(version, input_size))
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    b = mb.Conv(x, 3, bneck_info[0][0], 3, 2, 'same')
    segm_features = []
    for idx, info in enumerate(bneck_info):
        b = bottleneck_block(mb, b, *info)
        if (idx + 1) in segm_feature_indices:
            segm_features.append(b)
        if segmentation and idx + 1 == segm_feature_indices[-1]:
            return mb, segm_features

    last_conv = mb.Conv(b, bneck_info[-1][2], num_outputs[0], 1, 1)
    pool = mb.GlobalPool(last_conv)
    out = mb.FC(pool, num_outputs[0], num_outputs[1])
    if classify:
        out = mb.FC(out, num_outputs[1], num_classess)
    return mb

def mobilenet_v3_small_minimal(input_size=224, num_classess = 1000, classify = False):
    bneck_info = [
            [16, 16, 16, False, 2, 3],
            [16, 72, 24, False, 2, 3],
            [24, 88, 24, False, 1, 3],
            [24, 96, 40, False, 2, 3],
            [40, 240, 40, False, 1, 3],
            [40, 240, 40, False, 1, 3],
            [40, 120, 48, False, 1, 3],
            [48, 144, 48, False, 1, 3],
            [48, 288, 96, False, 2, 3],
            [96, 576, 96, False, 1, 3],
            [96, 576, 96, False, 1, 3],
            ]
    return mobilenet_v3('small_minimal', input_size, [576, 1024], num_classess, bneck_info, classify)

def mobilenet_v3_large(input_size=224, num_classess = 1000, classify = False, *args, **kwargs):
    bneck_info = [
            [16, 16, 16, False, 1, 3],
            [16, 64, 24, False, 2, 3],
            [24, 72, 24, False, 1, 3],
            [24, 72, 40, True, 2, 5],
            [40, 120, 40, True, 1, 5],
            [40, 120, 40, True, 1, 5],
            [40, 240, 80, False, 2, 3],
            [80, 200, 80, False, 1, 3],
            [80, 184, 80, False, 1, 3],
            [80, 184, 80, False, 1, 3],
            [80, 480, 112, True, 1, 3],
            [112, 672, 112, True, 1, 3],
            [112, 672, 160, True, 2, 5],
            [160, 960, 160, True, 1, 5],
            [160, 960, 160, True, 1, 5],
            ]
    return mobilenet_v3('large', input_size, [960, 1280], num_classess, bneck_info, classify, *args, **kwargs)

def mobilenet_v3_large_minimal(input_size=224, num_classess = 1000, classify = False):
    bneck_info = [
            [16, 16, 16, False, 1, 3],
            [16, 64, 24, False, 2, 3],
            [24, 72, 24, False, 1, 3],
            [24, 72, 40, False, 2, 3],
            [40, 120, 40, False, 1, 3],
            [40, 120, 40, False, 1, 3],
            [40, 240, 80, False, 2, 3],
            [80, 200, 80, False, 1, 3],
            [80, 184, 80, False, 1, 3],
            [80, 184, 80, False, 1, 3],
            [80, 480, 112, False, 1, 3],
            [112, 672, 112, False, 1, 3],
            [112, 672, 160, False, 2, 3],
            [160, 960, 160, False, 1, 3],
            [160, 960, 160, False, 1, 3],
            ]
    return mobilenet_v3('large_minimal', input_size, [960, 1280], num_classess, bneck_info, classify)

def mobilenet_v3_edgetpu(
        input_size=224,
        num_classess=1000,
        classify = False,
        *args,
        **kwargs
        ):
    bneck_info = [
            [32, 16, 16, False, 1, 3, True],
            [16, 128, 32, False, 2, 3, True],
            [32, 128, 32, False, 1, 3, True],
            [32, 128, 32, False, 1, 3, True],
            [32, 128, 32, False, 1, 3, True],
            [32, 256, 48, False, 2, 3, True],
            [48, 192, 48, False, 1, 3, True],
            [48, 192, 48, False, 1, 3, True],
            [48, 192, 48, False, 1, 3, True],
            [48, 48*8, 96, False, 2, 3],
            [96, 96*4, 96, False, 1, 3],
            [96, 96*4, 96, False, 1, 3],
            [96, 96*4, 96, False, 1, 3],
            [96, 96*8, 96, False, 1, 3, False, False],
            [96, 96*4, 96, False, 1, 3],
            [96, 96*4, 96, False, 1, 3],
            [96, 96*4, 96, False, 1, 3],
            [96, 96*8, 160, False, 2, 5],
            [160, 160*4, 160, False, 1, 5],
            [160, 160*4, 160, False, 1, 5],
            [160, 160*4, 160, False, 1, 5],
            [160, 160*8, 192, False, 1, 3],
            ]
    return mobilenet_v3('edgeTPU', input_size, [1280, 1280], num_classess, bneck_info, classify, *args, **kwargs)


def inverted_residual(mb : ModelBuilder, x, inp, oup, stride, expand_ratio, dilation = 1):
    hidden_dim = int(round(inp * expand_ratio))
    use_res_connect = stride == 1 and inp == oup
    in_x = x
    if expand_ratio != 1:
        x = mb.Conv(x, inp, hidden_dim, 1)
    x = mb.DWConv(x, hidden_dim, 3, stride, 'same', dilation=dilation)
    x = mb.Conv(x, hidden_dim, oup, 1, 1, 0, activation='Linear')
    if use_res_connect:
        x = mb.Sum(in_x, x)
    return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def mobilenet_v2(
        input_size=224,
        num_classess=1000,
        width_mult=1.0,
        output_stride=32,
        inverted_residual_setting=None,
        round_nearest=8,
        softmax=False,
        backbone = False):
    input_channel = 32
    last_channel = 1280
    if inverted_residual_setting is None:
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

    if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
        raise ValueError(
            "inverted_residual_setting should be non-empty or a 4-element listm got {}".format(inverted_residual_setting))

    last_channel = _make_divisible(
        last_channel * max(1.0, width_mult), round_nearest)
    mb = ModelBuilder("MobilenetV2_{}".format(input_size))
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    x = mb.Conv(x, 3, input_channel, 3, 2, 1)
    ids = [x]
    current_stride = 2
    rate = 1
    for t, c, n, s in inverted_residual_setting:
        output_channel = _make_divisible(c * width_mult, round_nearest)
        if current_stride == output_stride:
            dilation = rate
            rate *= s
            s = 1
        else:
            dilation = 1
            current_stride *= s
        for i in range(n):
            stride = s if i == 0 else 1
            x = inverted_residual(mb, x, input_channel,
                                  output_channel, stride, expand_ratio=t, dilation=dilation)
            input_channel = output_channel
            ids.append(x)
    if backbone:
        return mb, [ids[3], ids[-1]]
    x = mb.Conv(x, input_channel, last_channel, 1, 1, 0)
    x = mb.GlobalPool(x)
    x = mb.FC(x, last_channel, num_classess)
    if softmax:
        x = mb.Softmax(x)
    return mb

def conv_dw(mb, x, inp, oup, stride):
    x = mb.DWConv(x, inp, 3, stride, 1)
    x = mb.Conv(x, inp, oup, 1, 1, 0)
    return x


def mobilenet(
        input_size=224,
        backbone = False,
        num_classess=1000,
        softmax=False):

    mb = ModelBuilder("MobilenetV1_{}".format(input_size))
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    x = mb.Conv(x, 3, 32, 3, 2, 1)
    x = conv_dw(mb, x, 32, 64, 1)
    x = conv_dw(mb, x, 64, 128, 2)
    x = conv_dw(mb, x, 128, 128, 1)
    x = conv_dw(mb, x, 128, 256, 2)
    x = conv_dw(mb, x, 256, 256, 1)
    x = conv_dw(mb, x, 256, 512, 2)
    x = conv_dw(mb, x, 512, 512, 1)
    x = conv_dw(mb, x, 512, 512, 1)
    x = conv_dw(mb, x, 512, 512, 1)
    x = conv_dw(mb, x, 512, 512, 1)
    x11 = conv_dw(mb, x, 512, 512, 1)
    x = conv_dw(mb, x11, 512, 1024, 2)
    x13 = conv_dw(mb, x, 1024, 1024, 1)
    if backbone:
        return mb, [x11, x13]
    x = mb.GlobalPool(x13)
    x = mb.FC(x, 1024, 1000)
    if softmax:
        x = mb.Softmax(x)
    return mb
