# Basic blocks


act_func = 'ReLU'


def Identity(mb, input, name=None):
    return mb.MaxPool(input, k=1, stride=1, pad=0, name=name)


def Conv(mb, input, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
    out = mb.Conv(input, c1, c2, k, stride=s, pad='same' if p is None else p, dilation=d, groups=g, activation=act_func if act else None)
    return out


def Concat(mb, input_list, dimension=1):
    if isinstance(input_list, str):
        input_list = [input_list]
    out = mb.Concat(input_list, axis=dimension)
    return out


def ELANCat(mb, input_list, dimension=1):
    dummy_layers = [Identity(mb, input, name=f'{input}_cat') for input in input_list[1:]]
    cat = mb.Concat([input_list[0]] + dummy_layers)
    return cat


def ConvSumCat(mb, input_list, c1s, c2, k=1, s=1, p=None, g=1, act=True):
    conv = [Conv(mb, input, c1, c2, k, s, p, g, act=False) for input, c1 in zip(input_list, c1s)]
    sum = conv[0]
    for layer in conv[1:-1]:
        sum = mb.Sum(sum, layer)
    sum = mb.Sum(sum, conv[-1], activation=act_func if act else None)
    return sum


def Upsample(mb, input, size=None, scale_factor=None, mode='nearest'):
    out = mb.Upsample(input, scale=scale_factor, size=None, algorithm=mode)
    return out


def Silence(mb, input):
    return input    # Bypass


def SP(mb, input, k=3, s=1):
    out = mb.MaxPool(input, k, stride=s, pad='same')
    return out


def MP(mb, input, k=2):
    out = mb.MaxPool(input, k, stride=k)
    return out


def ADown(mb, input, c1, c2):
    c = c2 // 2
    x = mb.AvgPool(input, 2, 1, 0)
    x1, x2 = mb.Split(x, (c1 // 2, c1 // 2))
    x1 = Conv(mb, x1, c1 // 2, c, 3, 2, 1)
    x2 = mb.MaxPool(x2, 3, 2, 1)
    x2 = Conv(mb, x2, c1 // 2, c, 1, 1, 0)
    cat = mb.Concat([x1, x2])
    return cat


def RepConvN(mb, input, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=True):
    assert k == 3 and p == 1
    return mb.Conv(input, c1, c2, k, s, p, groups=g, dilation=d, bias=True, activation=act_func if act else None)


def RepNBottleneck(mb, input, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
    c_ = int(c2 * e)
    add = shortcut and c1 == c2
    cv1 = RepConvN(mb, input, c1, c_, k[0], 1)
    cv2 = Conv(mb, cv1, c_, c2, k[1], 1, g=g)
    if add:
        out = mb.Sum(input, cv2)
    else:
        out = cv2
    return out


def RepNCSP(mb, input, c1, c2, n=1, shortcut=True, g=1, e=0.5):
    c_ = int(c2 * e)
    cv1 = Conv(mb, input, c1, c_, 1, 1)
    m = cv1
    for _ in range(n):
        m = RepNBottleneck(mb, m, c_, c_, shortcut, g, e=1.0)
    cv2 = Conv(mb, input, c1, c_, 1, 1)
    cat = mb.Concat([m, cv2])
    cv3 = Conv(mb, cat, 2 * c_, c2, 1)
    return cv3


def RepNCSPELAN4(mb, input, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
    c = c3 // 2
    cv1 = [Conv(mb, input, c1, c, 1, 1) for _ in range(2)]
    cv2 = RepNCSP(mb, cv1[-1], c, c4, c5)
    cv2 = Conv(mb, cv2, c4, c4, 3, 1)
    cv3 = RepNCSP(mb, cv2, c4, c4, c5)
    cv3 = Conv(mb, cv3, c4, c4, 3, 1)
    #cv4 = ConvSumCat(mb, [cv3, cv2, cv1[1], cv1[0]], [c4, c4, c, c], c2, 1, 1)
    cat = ELANCat(mb, [cv3, cv2, cv1[1], cv1[0]])
    cv4 = Conv(mb, cat, c3+2*c4, c2, 1, 1)
    return cv4


def CatRepNCSPELAN4(mb, inputs, c1s, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
    c = c3 // 2
    cv1_0 = Conv(mb, inputs[0], c1s[0], c, 1, 1, act=False)
    cv1_1 = Conv(mb, inputs[0], c1s[0], c, 1, 1, act=False)
    for input, c1 in zip(inputs[1:], c1s[1:]):
        t0 = Conv(mb, input, c1, c, 1, 1, act=False)
        cv1_0 = mb.Sum(cv1_0, t0, activation=(act_func if input == inputs[-1] else None))
        t1 = Conv(mb, input, c1, c, 1, 1, act=False)
        cv1_1 = mb.Sum(cv1_1, t1, activation=(act_func if input == inputs[-1] else None))
    cv2 = RepNCSP(mb, cv1_1, c, c4, c5)
    cv2 = Conv(mb, cv2, c4, c4, 3, 1)
    cv3 = RepNCSP(mb, cv2, c4, c4, c5)
    cv3 = Conv(mb, cv3, c4, c4, 3, 1)
    #cv4 = ConvSumCat(mb, [cv3, cv2, cv1_1, cv1_0], [c4, c4, c, c], c2, 1, 1)
    cat = ELANCat(mb, [cv3, cv2, cv1_1, cv1_0])
    cv4 = Conv(mb, cat, c3+2*c4, c2, 1, 1)
    return cv4


def SPPELAN(mb, input, c1, c2, c3): # ch_in, ch_out, number, shortcut, groups, expansion
    c = c3
    cv1 = Conv(mb, input, c1, c3, 1, 1)
    cv2 = SP(mb, cv1, 5)
    cv3 = SP(mb, cv2, 5)
    cv4 = SP(mb, cv3, 5)
    cv5 = ConvSumCat(mb, [cv4, cv3, cv2, cv1], [c3, c3, c3, c3], c2, 1, 1)
    return cv5


def CBLinear(mb, input, c1, c2s, k=1, s=1, p=None, g=1):
    outs = []
    for c2 in c2s:
        outs.append(mb.Conv(input, c1, c2, k, s, 'same' if p is None else p, g, bias=True, activation=None))
    return outs


def CBFuse(mb, inputs, idx):
    input_names = [x[idx[i]] for i, x in enumerate(inputs[:-1])] + inputs[-1:]
    input_tensors = [mb.model_dict[name].output for name in input_names]
    target_size = input_tensors[-1].shape[-2:]  # H, W
    out = input_names[-1]
    for name, tensor in zip(input_names[:-1], input_tensors[:-1]):
        if tensor.shape[-2:] != target_size:
            name = mb.Upsample(name, size=target_size)
            name = Identity(mb, name, name=f'{name}_out')
        out = mb.Sum(out, name)
    return out


def DualDDetect(mb, inputs, nc=80, ch=(), inplace=True):
    return inputs   # offload
