from models import ModelBuilder
from functools import partial
import inspect
from collections import OrderedDict


def get_kwargs():
    """
    Gets kwargs of given called function.
    You can use this instead "local()"
    got idea from https://stackoverflow.com/questions/582056/getting-list-of-parameter-names-inside-python-function
    """
    frame = inspect.stack()[1][0]
    varnames, _, _, values = inspect.getargvalues(frame)

    called_from_class_method = varnames[0] == "self"
    if called_from_class_method:
        varnames = varnames[1:]

    kwargs = {i: values[i] for i in varnames}
    return kwargs


def iter_to_number(x):
    if isinstance(x, list) or isinstance(x, tuple):
        assert len(x) == 2 and x[0] == x[1]
        return x[0]
    else:
        return x


mb = ModelBuilder()


def reset_mb():
    global mb
    mb.model_dict = OrderedDict()
    mb.name_gen = {}


def set_input_tensor(tensor_shape):
    global mb
    return mb.set_input_tensor(tensor_shape=tensor_shape)


class Module:
    def __init__(self):
        self._modules = OrderedDict()

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError

    def add_module(self, name, module):
        self._modules[name] = module

    def __getattr__(self, name):
        if name in self._modules.keys():
            return self._modules[name]
        else:
            raise ValueError


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, x):
        for op in self._modules.values():
            x = op(x)
        return x

    def append(self, module):
        self.add_module(str(len(self._modules)), module)


class ModuleList(Sequential):
    def __add__(self, module):
        self.append(module)


class ModuleDict(Module):
    def items(self):
        return self._modules.items()


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        if bias == False:
            # print("I'm sorry I will fix bias to True...")
            bias = True
        self.conv2d = partial(
            mb.Conv,
            in_c=in_channels,
            out_c=out_channels,
            k=kernel_size,
            stride=iter_to_number(stride),
            pad=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv2d(x)


def max_pool2d(x, kernel_size, stride=1, padding=0):
    return mb.MaxPool(x, kernel_size, iter_to_number(stride), padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        kwargs = get_kwargs()
        self.pool = partial(max_pool2d, **kwargs)

    def forward(self, x):
        return self.pool(x)


def avg_pool2d(x, kernel_size, stride=1, padding=0):
    return mb.AvgPool(x, kernel_size, iter_to_number(stride), padding)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        kwargs = get_kwargs()
        self.pool = partial(avg_pool2d, **kwargs)

    def forward(self, x):
        return self.pool(x)


def adaptive_avg_pool2d(x, output_size):
    assert iter_to_number(output_size) == 1
    return mb.GlobalPool(x)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        self.pool = partial(adaptive_avg_pool2d, output_size=output_size)

    def forward(self, x):
        return self.pool(x)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.fc = partial(mb.FC, in_c=in_features, out_c=out_features, bias=bias)

    def forward(self, x):
        return self.fc(x)


def cat(ops, dim):
    assert dim == 1
    return mb.Concat(ops)


def add(input1, input2):
    return mb.Sum(input1, input2)


""" Ops below are ignored"""


class IgnoredOps(Module):
    def forward(self, input):
        return input


class BatchNorm2d(IgnoredOps):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        pass


class ReLU(IgnoredOps):
    def __init__(self, inplace=False):
        pass


class ReLU6(ReLU):
    pass


def relu(x, inplace=True):
    return x


def relu6(x, inplace=True):
    return x


class Dropout(IgnoredOps):
    def __init__(self, p=0.5, inplace=False):
        pass


def dropout(x, training, p=0):
    return x
