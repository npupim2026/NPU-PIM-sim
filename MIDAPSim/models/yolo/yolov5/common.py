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


def Upsample(mb, input, size=None, scale_factor=None, mode='nearest'):
    out = mb.Upsample(input, scale=scale_factor, size=None, algorithm=mode)
    return out


def Silence(mb, input):
    return input    # Bypass


def BottleNeck(mb, input, c1, c2, shortcut=True, g=1, e=0.5):
    c_ = int(c2 * e)
    cv1 = Conv(mb, input, c1, c_, 1, 1)
    cv2 = Conv(mb, cv1, c_, c2, 3, 1, g=g)
    if shortcut and c1 == c2:
        out = mb.Sum(input, cv2)
    else:
        out = cv2
    return out


def C3(mb, input, c1, c2, n=1, shortcut=True, g=1, e=0.5):
    c_ = int(c2 * e)
    cv1 = Conv(mb, input, c1, c_, 1, 1)
    m = cv1
    for _ in range(n):
        m = BottleNeck(mb, m, c_, c_, shortcut, g, e=1.0)
    cv2 = Conv(mb, input, c1, c_, 1, 1)
    if c_ % 64 != 0:
        cv3_1 = Conv(mb, m, c_, c2, 1)
        cv3_2 = Conv(mb, cv2, c_, c2, 1)
        cv3 = mb.Sum(cv3_1, cv3_2)
    else:
        cat = mb.Concat([m, cv2])
        cv3 = Conv(mb, cat, 2 * c_, c2, 1)
    return cv3


def CatC3(mb, inputs, c1s, c2, n=1, shortcut=True, g=1, e=0.5):
    c_ = int(c2 * e)
    cv1_0 = Conv(mb, inputs[0], c1s[0], c_, 1, 1, act=False)
    cv1_1 = Conv(mb, inputs[1], c1s[1], c_, 1, 1, act=False)
    cv2_0 = Conv(mb, inputs[0], c1s[0], c_, 1, 1, act=False)
    cv2_1 = Conv(mb, inputs[1], c1s[1], c_, 1, 1, act=False)
    cv1 = mb.Sum(cv1_0, cv1_1, activation=act_func)
    cv2 = mb.Sum(cv2_0, cv2_1, activation=act_func)
    m = cv1
    for _ in range(n):
        m = BottleNeck(mb, m, c_, c_, shortcut, g, e=1.0)
    if c_ % 64 != 0:
        cv3_1 = Conv(mb, m, c_, c2, 1)
        cv3_2 = Conv(mb, cv2, c_, c2, 1)
        cv3 = mb.Sum(cv3_1, cv3_2)
    else:
        cat = mb.Concat([m, cv2])
        cv3 = Conv(mb, cat, 2 * c_, c2, 1)
    return cv3


def SPPF(mb, input, c1, c2, k=5):
    c_ = c1 // 2
    cv1 = Conv(mb, input, c1, c_, 1, 1)
    y1 = mb.MaxPool(cv1, k=k, stride=1, pad='same')
    y2 = mb.MaxPool(y1, k=k, stride=1, pad='same')
    y3 = mb.MaxPool(y2, k=k, stride=1, pad='same')
    cat = ELANCat(mb, [y3, y2, y1, cv1])
    cv2 = Conv(mb, cat, 4 * c_, c2, 1, 1)
    return cv2
