import os
import yaml
import math
from models.model_builder import ModelBuilder
from .common import Conv, Concat, Upsample, Silence, C3, CatC3, SPPF


def parse_model(mb, input, in_c, num_classes=None, cfg_path=os.path.join(os.path.dirname(__file__), 'cfg', 'yolov5s.yaml')):
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    if num_classes is None:
        num_classes = cfg['nc']
    gd = cfg['depth_multiple']
    gw = cfg['width_multiple']
    ch_mul = 8

    layers = [input]
    out_c = [in_c]
    for i, (f, n, m, args) in enumerate(cfg['backbone'] + cfg['head']):
        if m == 'nn.Upsample':
            m = 'Upsample'
        elif m == 'Detect':
            continue
        m = eval(m)
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, SPPF]:
            args = [mb, layers[f], out_c[f]] + [math.ceil(args[0] * gw / ch_mul) * ch_mul] + args[1:]
        elif m is C3:
            args = [mb, layers[f], out_c[f]] + [math.ceil(args[0] * gw / ch_mul) * ch_mul, n] + args[1:]
            n = 1
        elif m is CatC3:
            args = [mb, list(map(lambda x: layers[x], f)), list(map(lambda x: out_c[x], f))] + [math.ceil(args[0] * gw / ch_mul) * ch_mul, n] + args[1:]
            n = 1
        else:
            args = [mb, layers[f] if isinstance(f, int) else list(map(lambda x: layers[x], f))] + args
        if i == 0:
            layers = []
            out_c = []
        new_layer = m(*args)
        new_out_c = mb.model_dict[new_layer]['output'].shape[1]
        if n > 1:
            for _ in range(n-1):
                args[1] = new_layer
                if m in [Conv, C3, SPPF]:
                    args[2] = new_out_c
                new_layer = m(*args)
                new_out_c = mb.model_dict[new_layer]['output'].shape[1]
        layers.append(new_layer)
        out_c.append(new_out_c)


def yolov5s(input_size=640, num_classes=80):
    cfg_path = os.path.join(os.path.dirname(__file__), 'cfg', 'yolov5s.yaml')
    mb = ModelBuilder(f'YOLOv5s_{input_size}')
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    parse_model(mb, x, 3, num_classes, cfg_path=cfg_path)
    return mb


def yolov5m(input_size=640, num_classes=80):
    cfg_path = os.path.join(os.path.dirname(__file__), 'cfg', 'yolov5m.yaml')
    mb = ModelBuilder(f'YOLOv5m_{input_size}')
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    parse_model(mb, x, 3, num_classes, cfg_path=cfg_path)
    return mb


def yolov5l(input_size=640, num_classes=80):
    cfg_path = os.path.join(os.path.dirname(__file__), 'cfg', 'yolov5l.yaml')
    mb = ModelBuilder(f'YOLOv5l_{input_size}')
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    parse_model(mb, x, 3, num_classes, cfg_path=cfg_path)
    return mb
