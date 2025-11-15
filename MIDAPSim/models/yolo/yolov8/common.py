# Basic Blocks


act_func = 'ReLU'


def Identity(mb, input, name=None):
    return mb.MaxPool(input, k=1, stride=1, pad=0, name=name)


def Conv(mb, input, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
    out = mb.Conv(input, c1, c2, k, stride=s, pad='same' if p is None else p, groups=g, dilation=d, activation=act_func if act else None)
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


def Bottleneck(mb, input, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
    c_ = int(c2 * e)
    cv1 = Conv(mb, input, c1, c_, k[0], 1)
    cv2 = Conv(mb, cv1, c_, c2, k[1], 1, g=g)
    if shortcut and c1 == c2:
        out = mb.Sum(input, cv2)
    else:
        out = cv2
    return out


def C2f(mb, input, c1, c2, n=1, shortcut=False, g=1, e=0.5):
    c = int(c2 * e)
    y = [Conv(mb, input, c1, c, 1, 1) for _ in range(2)]
    for _ in range(n):
        y.append(Bottleneck(mb, y[-1], c, c, shortcut, g, k=((3, 3), (3, 3)), e=1.0))
    cv2 = ConvSumCat(mb, list(reversed(y)), [c] * (n+2), c2, 1)
    return cv2


def CatC2f(mb, inputs, c1s, c2, n=1, shortcut=False, g=1, e=0.5):
    c = int(c2 * e)
    cv0_0 = Conv(mb, inputs[0], c1s[0], c, 1, 1, act=False)
    cv0_1 = Conv(mb, inputs[1], c1s[1], c, 1, 1, act=False)
    cv1_0 = Conv(mb, inputs[0], c1s[0], c, 1, 1, act=False)
    cv1_1 = Conv(mb, inputs[1], c1s[1], c, 1, 1, act=False)
    y = [mb.Sum(cv0_0, cv0_1, activation=act_func), mb.Sum(cv1_0, cv1_1, activation=act_func)]
    for _ in range(n):
        y.append(Bottleneck(mb, y[-1], c, c, shortcut, g, k=((3, 3), (3, 3)), e=1.0))
    cv2 = ConvSumCat(mb, list(reversed(y)), [c] * (n+2), c2, 1)
    return cv2


def SPPF(mb, input, c1, c2, k=5):
    c_ = c1 // 2
    cv1 = Conv(mb, input, c1, c_, 1, 1)
    y1 = mb.MaxPool(cv1, k=k, stride=1, pad='same')
    y2 = mb.MaxPool(y1, k=k, stride=1, pad='same')
    y3 = mb.MaxPool(y2, k=k, stride=1, pad='same')
    cat = ELANCat(mb, [y3, y2, y1, cv1])
    cv2 = Conv(mb, cat, 4 * c_, c2, 1, 1)
    return cv2
