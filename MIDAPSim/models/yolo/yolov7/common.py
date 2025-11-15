# Basic Blocks


act_func = 'ReLU'


def MP(mb, input, k=2):
    out = mb.MaxPool(input, k, stride=k)
    return out


def SP(mb, input, k=3, s=1):
    out = mb.MaxPool(input, k, stride=s, pad='same')
    return out


def Identity(mb, input, name=None):
    return mb.MaxPool(input, k=1, stride=1, pad=0, name=name)


def Conv(mb, input, c1, c2, k=1, s=1, p=None, g=1, act=True):
    out = mb.Conv(input, c1, c2, k, stride=s, pad='same' if p is None else p, groups=g, activation=act_func if act else None)
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


def SPPCSPC(mb, input, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
    c_ = int(2 * c2 * e)
    cv1 = Conv(mb, input, c1, c_, 1, 1)
    cv2 = Conv(mb, input, c1, c_, 1, 1)
    cv3 = Conv(mb, cv1, c_, c_, 3, 1)
    cv4 = Conv(mb, cv3, c_, c_, 1, 1)
    cat_list = [cv4] + [mb.MaxPool(cv4, x, stride=1, pad='same') for x in k]
    cv5 = ConvSumCat(mb, cat_list, [c_ for _ in range(len(cat_list))], c_, 1, 1)
    cv6 = Conv(mb, cv5, c_, c_, 3, 1)
    cv7_1 = Conv(mb, cv4, c_, c2, 1, 1, act=False)
    cv7_2 = Conv(mb, cv6, c_, c2, 1, 1, act=False)
    cv7 = mb.Sum(cv7_1, cv7_2, activation=act_func)
    return cv7


def RepConv(mb, input, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=True):
    out = mb.Conv(input, c1, c2, k, stride=s, pad='same' if p is None else p, groups=g, activation=act_func if act else None)
    return out


def Detect(mb, input, nc=80, anchors=(), ch=()):
    return input    # offload
